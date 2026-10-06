"""
Policy runtime check: is a policy fast enough to enter a tournament?

    python check_policy.py policies.attacker.my_atk --team red
    python check_policy.py policies.attacker.my_atk --team red --quick       # 1 game per size
    python check_policy.py policies.attacker.my_atk --team red --calibrate   # measure only

Plays the policy over every config under config/example_configs/, times each
strategy call, and writes an HTML report. Timings are compared in machine
units — multiples of the time this machine needs for a fixed reference
workload — so the same budget file applies on a laptop and on a server.

Exit code: 0 pass, 1 fail. The limits and what fails a policy: README, section 7.
"""

import argparse
import importlib
import logging
import os
import platform
import random
import re
import signal
import sys
import time
import types
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from lib.core.console import LogLevel  # noqa: E402
from lib.game.game_engine import GameEngine  # noqa: E402
from lib.utils.perf_report import fmt_time, fmt_units, write_report  # noqa: E402

EXTRA_DEFS = "config/game_config.yml"
DEFAULT_OPPONENT = {"red": "example.example_def", "blue": "example.example_atk"}

# A timed call is (agent name, tick, seconds).
Call = Tuple[str, int, float]


# --------------------------- Machine unit ---------------------------------

REFERENCE_NODES = 15
REFERENCE_REPEATS = 8


def _held_karp(n: int) -> int:
    """Exact TSP on a fixed n-node instance. Pure Python and dict-heavy, like strategy code."""
    d = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d[i][j] = d[j][i] = (i * 37 + j * 91 + i * j * 13) % 97 + 1

    best = {(1 << j, j): d[0][j] for j in range(1, n)}
    full = (1 << n) - 2
    for mask in range(2, full + 1, 2):
        if mask & (mask - 1) == 0:
            continue
        for j in range(1, n):
            bit = 1 << j
            if not mask & bit:
                continue
            prev = mask ^ bit
            cost = None
            for k in range(1, n):
                if prev >> k & 1:
                    c = best[(prev, k)] + d[k][j]
                    if cost is None or c < cost:
                        cost = c
            best[(mask, j)] = cost
    return min(best[(full, j)] + d[j][0] for j in range(1, n))


def measure_machine_unit(runs: int = 5) -> float:
    """Seconds this machine needs for the reference workload: the fastest of `runs`."""
    timings = []
    for _ in range(runs):
        t0 = time.perf_counter()
        for _ in range(REFERENCE_REPEATS):
            _held_karp(REFERENCE_NODES)
        timings.append(time.perf_counter() - t0)
    return min(timings)


# --------------------------- Timing a game --------------------------------

class CheckAbort(BaseException):
    """Ends the whole check. Not an Exception, so the engine's own handlers let it through."""

    calls: List[Call] = []


def _timed(fn: Any, agent: str, calls: List[Call], abort_after: float, unit: float) -> Any:
    def strategy(state: dict) -> str:
        tick = state["time"]
        t0 = time.perf_counter()
        try:
            result = fn(state)
        except Exception as exc:
            raise CheckAbort(f"strategy raised {type(exc).__name__}: {exc} ({agent}, tick {tick})") from exc
        dt = time.perf_counter() - t0
        calls.append((agent, tick, dt))
        # The engine rejects these too, but quietly: the agent just stands still.
        if not isinstance(result, str):
            raise CheckAbort(f"strategy returned {type(result).__name__}, not str ({agent}, tick {tick})")
        if not isinstance(state.get("action"), int):
            raise CheckAbort(f"strategy did not set state['action'] to an int ({agent}, tick {tick})")
        if dt > abort_after:
            raise CheckAbort(f"one call took {fmt_units(dt / unit)} units, {fmt_time(dt)} ({agent}, tick {tick})")
        return result

    return strategy


def _timed_policy(module: Any, calls: List[Call], abort_after: float, unit: float) -> Any:
    """Stand-in for the policy module whose strategies record their own call times."""

    def map_strategy(agent_config: dict) -> dict:
        return {name: _timed(fn, name, calls, abort_after, unit) for name, fn in module.map_strategy(agent_config).items()}

    return types.SimpleNamespace(__name__=module.__name__, map_strategy=map_strategy)


def _on_alarm(signum: int, frame: Any) -> None:
    raise CheckAbort("a game ran past its time limit")


def play(
    config: Path, seed: int, policy: Any, opponent: str, team: str, unit: float, abort_after: float, timeout: float
) -> Tuple[List[Call], Dict[str, float]]:
    """Play one headless game and return the policy's timed calls and the game's payoff."""
    calls: List[Call] = []
    timed = _timed_policy(policy, calls, abort_after, unit)
    red, blue = (timed, opponent) if team == "red" else (opponent, timed)

    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    # A call that never returns can only be stopped from outside it. SIGALRM does that on
    # macOS and Linux; Windows has no equivalent, and there Ctrl-C is the way out.
    can_alarm = hasattr(signal, "SIGALRM")
    if can_alarm:
        signal.signal(signal.SIGALRM, _on_alarm)
        signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        engine = GameEngine.launch_from_files(
            config_main=str(config),
            extra_defs=EXTRA_DEFS,
            red_strategy=red,
            blue_strategy=blue,
            log_name=None,
            set_level=LogLevel.ERROR,
            vis=False,
        )
    except CheckAbort as abort:
        abort.calls = calls
        raise
    finally:
        if can_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
    # Red's components; the payoff model is zero-sum, so blue's are their negatives.
    payoff = {
        "team": engine.red_payoff_accum if team == "red" else engine.blue_payoff_accum,
        "capture": engine.red_capture_reward,
        "tag": engine.red_tag_penalty,
        "discover": engine.red_discover_reward,
    }
    return calls, payoff


def summarise_game(config: Path, seed: int, calls: List[Call], payoff: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Split a game's calls into each agent's first call (one-time setup) and the rest."""
    seen: set = set()
    first: List[Call] = []
    steady: List[Call] = []
    for call in calls:
        (steady if call[0] in seen else first).append(call)
        seen.add(call[0])
    return {
        "config": config,
        "size": config.parent.name,
        "seed": seed,
        "calls": calls,
        "payoff": payoff,
        "steady_n": len(steady),
        "steady_sum": sum(c[2] for c in steady),
        "max": max(steady, key=lambda c: c[2], default=None),
        "first": max(first, key=lambda c: c[2], default=None),
    }


# --------------------------- Budgets and verdict --------------------------

def load_budget(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def budget_for(budget: Dict[str, Any], size: str) -> Dict[str, float]:
    return {**budget["default"], **(budget.get("sizes") or {}).get(size, {})}


def _where(game: Dict[str, Any], call: Call) -> str:
    return f"{call[0]}, tick {call[1]}, {game['config'].stem}"


def summarise_size(size: str, games: List[Dict[str, Any]], limits: Dict[str, float], unit: float) -> Dict[str, Any]:
    steady_n = sum(g["steady_n"] for g in games)
    mean = sum(g["steady_sum"] for g in games) / steady_n if steady_n else 0.0
    worst = max((g for g in games if g["max"]), key=lambda g: g["max"][2], default=None)
    first = max((g for g in games if g["first"]), key=lambda g: g["first"][2], default=None)
    payoffs = [g["payoff"]["team"] for g in games if g["payoff"]]
    row = {
        "name": size,
        "games": len(games),
        "calls": sum(len(g["calls"]) for g in games),
        "budget": limits,
        "mean_s": mean,
        "max_s": worst["max"][2] if worst else 0.0,
        "max_where": _where(worst, worst["max"]) if worst else "",
        "first_s": first["first"][2] if first else 0.0,
        "first_where": _where(first, first["first"]) if first else "",
        "payoff": sum(payoffs) / len(payoffs) if payoffs else None,
    }
    row["fails"] = [
        label
        for label, key, limit in (
            ("mean per call", "mean_s", limits["mean_per_call"]),
            ("slowest call", "max_s", limits["max_per_call"]),
            ("first call", "first_s", limits["first_call"]),
        )
        if row[key] / unit > limit
    ]
    return row


def growth_notes(rows: List[Dict[str, Any]], team: str, unit: float) -> Tuple[List[str], List[str]]:
    """Compare mean call time between the smallest and largest team for each flag set."""
    notes: List[str] = []
    warnings: List[str] = []
    by_flags: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    for row in rows:
        m = re.match(r"R(\d+)B(\d+)(.*)", row["name"])
        if m:
            by_flags.setdefault(m.group(3), []).append((int(m.group(1 if team == "red" else 2)), row))
    for group in by_flags.values():
        (n_lo, lo), (n_hi, hi) = min(group, key=lambda x: x[0]), max(group, key=lambda x: x[0])
        if n_hi == n_lo or not lo["mean_s"]:
            continue
        ratio = hi["mean_s"] / lo["mean_s"]
        text = f"{lo['name']} → {hi['name']}: mean call time ×{ratio:.1f} for ×{n_hi / n_lo:.0f} agents"
        notes.append(text)
        # Per-call cost growing faster than the team does. Ignored while the policy is still
        # far inside its budget, where the ratio is mostly timer noise.
        if ratio > n_hi / n_lo and hi["mean_s"] / unit > 0.05 * hi["budget"]["mean_per_call"]:
            warnings.append(text + " — super-linear")
    return notes, warnings


# --------------------------- Main -----------------------------------------

def find_configs(config_dir: Path, quick: bool) -> List[Path]:
    by_size: Dict[str, List[Path]] = {}
    for path in sorted(config_dir.rglob("*.yml")):
        by_size.setdefault(path.parent.name, []).append(path)
    return [p for paths in by_size.values() for p in (paths[:1] if quick else paths)]


def print_table(rows: List[Dict[str, Any]], unit: float, judged: bool) -> None:
    print(f"\nmachine unit = {unit:.3f} s   (times below are multiples of it)")
    print(f"{'size':<12}{'calls':>7}{'mean/call':>12}{'max/call':>12}{'first-call':>12}{'payoff':>9}   result")
    for row in rows:
        result = "" if not judged else "PASS" if not row["fails"] else "FAIL  (" + "; ".join(row["fails"]) + ")"
        print(
            f"{row['name']:<12}{row['calls']:>7}{row['mean_s'] / unit:>12.5f}"
            f"{row['max_s'] / unit:>12.5f}{row['first_s'] / unit:>12.5f}"
            f"{'' if row['payoff'] is None else format(row['payoff'], '+.2f'):>9}   {result}"
        )
    limits = rows[0]["budget"] if rows else {}
    if limits:
        print(f"{'budget':<12}{'':>7}{limits['mean_per_call']:>12.5f}{limits['max_per_call']:>12.5f}{limits['first_call']:>12.5f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check that a policy is fast enough for a tournament.")
    parser.add_argument("policy", help="import path of the policy, e.g. policies.attacker.my_atk")
    parser.add_argument("--team", required=True, choices=["red", "blue"], help="team the policy plays")
    parser.add_argument("--opponent", help="import path of the opponent policy (default: the example policy)")
    parser.add_argument("--quick", action="store_true", help="play one game per size instead of all")
    parser.add_argument("--calibrate", action="store_true", help="measure and report only; no pass/fail")
    parser.add_argument("--configs", default="config/example_configs", help="directory of game configs")
    parser.add_argument("--budget", default="config/perf_budget.yml", help="budget file")
    parser.add_argument("--out", help="report path (default: reports/<policy>_<team>.html)")
    args = parser.parse_args()

    logging.disable(logging.INFO)  # gamms announces its log level once per game
    budget = load_budget(Path(args.budget))
    opponent = args.opponent or DEFAULT_OPPONENT[args.team]
    policy = importlib.import_module(args.policy)
    configs = find_configs(Path(args.configs), args.quick)
    if not configs:
        print(f"No config files found under {args.configs}")
        return 1

    print("Measuring this machine ...")
    unit = measure_machine_unit()
    abort_after = budget["abort_call"] * unit
    timeout = budget["game_timeout"] * unit
    print(f"machine unit = {unit:.3f} s")

    # One untimed game, so the shared shortest-path and visibility tables are built before timing starts.
    games: List[Dict[str, Any]] = []
    reasons: List[str] = []
    warnings: List[str] = []
    stopped = False
    try:
        play(configs[0], 0, policy, opponent, args.team, unit, abort_after, timeout)
        for i, config in enumerate(configs):
            print(f"\r[{i + 1}/{len(configs)}] {config.stem:<24}", end="", flush=True)
            games.append(summarise_game(config, i, *play(config, i, policy, opponent, args.team, unit, abort_after, timeout)))

    except CheckAbort as abort:
        stopped = True
        reasons.append(f"stopped early — {abort}")
        if abort.calls:
            games.append(summarise_game(configs[min(len(games), len(configs) - 1)], len(games), abort.calls))
    print("\r" + " " * 48 + "\r", end="")

    unit_end = measure_machine_unit(runs=3)
    if abs(unit_end - unit) / unit > 0.2:
        warnings.append(f"machine speed drifted during the run ({unit:.3f} s → {unit_end:.3f} s); timings are unreliable — close other programs and rerun")

    sizes = list(dict.fromkeys(g["size"] for g in games))
    rows = [summarise_size(s, [g for g in games if g["size"] == s], budget_for(budget, s), unit) for s in sizes]
    if args.calibrate:
        for row in rows:
            row["fails"] = []
    for row in rows:
        reasons.extend(f"{row['name']}: {label} over budget" for label in row["fails"])
    if stopped and rows:
        rows[-1]["fails"].append("stopped early")  # so the size it stopped in does not read as a pass
    growth, growth_warn = growth_notes(rows, args.team, unit)
    warnings.extend(growth_warn)

    slowest = max((g for g in games if g["max"]), key=lambda g: g["max"][2], default=None)
    timeline = None
    if slowest:
        per_tick: Dict[int, Call] = {}
        for call in slowest["calls"]:
            if call[1] not in per_tick or call[2] > per_tick[call[1]][2]:
                per_tick[call[1]] = call
        timeline = {
            "config": slowest["config"].stem,
            "peak": (slowest["max"][1], slowest["max"][2], slowest["max"][0]),
            "points": [(tick, per_tick[tick][2], per_tick[tick][0]) for tick in sorted(per_tick)],
            "budget_s": budget_for(budget, slowest["size"])["max_per_call"] * unit,
        }

    played = [g["payoff"] for g in games if g["payoff"]]
    payoff = {k: sum(p[k] for p in played) / len(played) for k in played[0]} if played else None

    verdict = "MEASURED" if args.calibrate else ("FAIL" if reasons else "PASS")
    print_table(rows, unit, not args.calibrate)
    for line in growth:
        print("growth  " + line)
    for line in warnings:
        print("WARNING " + line)
    for line in reasons:
        print("FAIL    " + line)

    out = Path(args.out) if args.out else ROOT / "reports" / f"{args.policy.rsplit('.', 1)[-1]}_{args.team}.html"
    write_report(
        {
            "policy": args.policy,
            "team": args.team,
            "opponent": opponent,
            "verdict": verdict,
            "reasons": reasons,
            "warnings": warnings,
            "growth": growth,
            "payoff": payoff,
            "unit": unit,
            "unit_end": unit_end,
            "quick": args.quick,
            "games": len(games),
            "rows": rows,
            "timeline": timeline,
            "python": platform.python_version(),
            "machine": f"{platform.system()} {platform.machine()}",
            "when": datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
        out,
    )
    print(f"\n{verdict}   report: {out}")
    return 1 if verdict == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
