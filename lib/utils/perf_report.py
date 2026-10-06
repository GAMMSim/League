"""HTML report for check_policy.py: one self-contained file, no external assets."""

import json
import math
from html import escape
from pathlib import Path
from typing import Any, Dict, List


def fmt_time(seconds: float) -> str:
    if seconds >= 1:
        return f"{seconds:.2f} s"
    ms = seconds * 1000
    if ms >= 10:
        return f"{ms:.0f} ms"
    if ms >= 0.1:
        return f"{ms:.2f} ms"
    return f"{ms:.3f} ms"


def fmt_units(units: float) -> str:
    """A time in machine units, to three significant figures."""
    if units == 0:
        return "0"
    if units >= 100:
        return f"{units:.0f}"
    return f"{units:.3g}"


def _fmt_pct(share: float) -> str:
    pct = share * 100
    if pct >= 10:
        return f"{pct:.0f}%"
    if pct >= 0.1:
        return f"{pct:.1f}%"
    return "<0.1%"


def _nice_ceiling(x: float) -> float:
    if x <= 0:
        return 1.0
    exp = 10 ** math.floor(math.log10(x))
    for m in (1, 2, 4, 6, 8, 10):  # each divides into four clean gridline steps
        if m * exp >= x:
            return m * exp
    return 10 * exp


_CSS = """
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --series: #2a78d6; --good: #0ca30c; --warning: #fab219; --critical: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --series: #3987e5;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1040px; margin: 0 auto; padding: 32px 16px 56px; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 15px; margin: 0 0 2px; }
.sub { color: var(--ink-2); margin: 0; }
.meta { color: var(--muted); font-size: 12px; margin: 4px 0 0; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 16px; margin-top: 16px; }
.verdict { display: flex; gap: 16px; align-items: flex-start; }
.verdict .badge { font-size: 48px; font-weight: 650; line-height: 1; white-space: nowrap; }
.verdict .mark { display: inline-block; width: 0.8em; text-align: center; }
.PASS .mark { color: var(--good); } .FAIL .mark { color: var(--critical); } .MEASURED .mark { color: var(--muted); }
.verdict ul { margin: 6px 0 0; padding-left: 18px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-top: 16px; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; }
.tile .label { color: var(--ink-2); font-size: 12px; }
.tile .value { font-size: 24px; font-weight: 600; }
.tile .note { color: var(--muted); font-size: 12px; }
.panels { display: grid; grid-template-columns: repeat(auto-fit, minmax(290px, 1fr)); gap: 16px; }
.panels .card { margin-top: 16px; }
.bars { margin-top: 12px; }
.bar-row { display: grid; grid-template-columns: 78px 1fr; align-items: center; gap: 8px; height: 26px; }
.bar-row .name { color: var(--ink-2); font-size: 12px; font-variant-numeric: tabular-nums; }
.track { position: relative; height: 26px; border-left: 1px solid var(--axis); margin-right: 88px; }  /* room for the value label */
.bar { position: absolute; left: 0; top: 6px; height: 14px; min-width: 2px; background: var(--series);
  border-radius: 0 4px 4px 0; }
.bar.over { background: var(--critical); }
.bar-row:hover .bar, .bar-row:focus .bar { filter: brightness(1.15); }
.bar-row:focus { outline: 1px solid var(--axis); outline-offset: 2px; }
.val { position: absolute; top: 4px; font-size: 12px; color: var(--ink); white-space: nowrap;
  background: var(--surface); padding: 0 2px;
  font-variant-numeric: tabular-nums; }
.limit { position: absolute; top: 0; bottom: 0; width: 1px; background: var(--ink-2); }
.axis-note { display: grid; grid-template-columns: 78px 1fr; gap: 8px; color: var(--muted); font-size: 11px; }
.axis-note .scale { position: relative; height: 16px; margin-right: 88px; }
.axis-note .scale span { position: absolute; top: 0; transform: translateX(-50%); white-space: nowrap; }
.axis-note .scale span:first-child { transform: none; }
svg { display: block; width: 100%; height: auto; margin-top: 8px; }
svg text { font: 11px system-ui, -apple-system, "Segoe UI", sans-serif; fill: var(--muted); }
svg .label { fill: var(--ink-2); }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; margin-top: 8px; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 6px 10px; border-bottom: 1px solid var(--grid); white-space: nowrap; }
th:first-child, td:first-child, th:last-child, td:last-child { text-align: left; }
th { color: var(--ink-2); font-weight: 500; font-size: 12px; }
td small { color: var(--muted); }
.tip { position: fixed; pointer-events: none; background: var(--ink); color: var(--surface);
  padding: 6px 8px; border-radius: 6px; font-size: 12px; display: none; z-index: 10; max-width: 280px; }
.tip b { display: block; font-size: 13px; }
.notes li { margin: 2px 0; }
"""

_JS = """
const tip = document.getElementById('tip');
function showTip(x, y, strong, rest) {
  tip.replaceChildren();
  const b = document.createElement('b'); b.textContent = strong; tip.appendChild(b);
  tip.appendChild(document.createTextNode(rest));
  tip.style.display = 'block';
  const w = tip.offsetWidth, h = tip.offsetHeight;
  tip.style.left = Math.max(8, Math.min(x + 12, innerWidth - w - 8)) + 'px';
  tip.style.top = Math.max(8, y - h - 12) + 'px';
}
function hideTip() { tip.style.display = 'none'; }
document.querySelectorAll('[data-strong]').forEach(el => {
  el.addEventListener('pointermove', e => showTip(e.clientX, e.clientY, el.dataset.strong, el.dataset.rest));
  el.addEventListener('pointerleave', hideTip);
  el.addEventListener('focus', () => { const r = el.getBoundingClientRect(); showTip(r.left + r.width / 2, r.top, el.dataset.strong, el.dataset.rest); });
  el.addEventListener('blur', hideTip);
});
const svg = document.getElementById('timeline');
if (svg) {
  const pts = JSON.parse(svg.dataset.points), hair = svg.querySelector('#hair'), dot = svg.querySelector('#dot');
  svg.addEventListener('pointermove', e => {
    const box = svg.getBoundingClientRect(), x = (e.clientX - box.left) * svg.viewBox.baseVal.width / box.width;
    let best = pts[0];
    for (const p of pts) if (Math.abs(p.x - x) < Math.abs(best.x - x)) best = p;
    hair.setAttribute('x1', best.x); hair.setAttribute('x2', best.x); hair.style.display = '';
    dot.setAttribute('cx', best.x); dot.setAttribute('cy', best.y); dot.style.display = '';
    showTip(e.clientX, e.clientY, best.t, 'tick ' + best.tick + ' · ' + best.agent);
  });
  svg.addEventListener('pointerleave', () => { hair.style.display = 'none'; dot.style.display = 'none'; hideTip(); });
}
"""


def _bar_panel(title: str, subtitle: str, rows: List[Dict[str, Any]], key: str, budget_key: str, unit: float) -> str:
    """Horizontal bars, one per size, as a share of that size's budget."""
    shares = [(r, r[key] / unit / r["budget"][budget_key]) for r in rows]
    scale = max(1.0, max((s for _, s in shares), default=0.0)) * 1.25
    limit_pos = 100 / scale
    out = [f'<div class="card"><h2>{escape(title)}</h2><p class="sub">{escape(subtitle)}</p><div class="bars">']
    for row, share in shares:
        width = share / scale * 100
        over = share > 1
        label = _fmt_pct(share) + (" ✕ over" if over else "")
        rest = f"{fmt_units(row[key] / unit)} units · budget {row['budget'][budget_key]} units · {fmt_time(row[key])} on this machine"
        out.append(
            f'<div class="bar-row" tabindex="0" data-strong="{escape(row["name"])}: {_fmt_pct(share)} of budget" data-rest="{escape(rest)}">'
            f'<span class="name">{escape(row["name"])}</span><div class="track">'
            f'<div class="limit" style="left:{limit_pos:.2f}%"></div>'
            f'<div class="bar{" over" if over else ""}" style="width:{width:.2f}%"></div>'
            f'<span class="val" style="left:calc({width:.2f}% + 6px)">{label}</span></div></div>'
        )
    out.append(
        f'</div><div class="axis-note"><span></span><div class="scale"><span style="left:0">0</span>'
        f'<span style="left:{limit_pos:.2f}%">budget (100%)</span></div></div></div>'
    )
    return "".join(out)


def _timeline(timeline: Dict[str, Any], unit: float) -> str:
    """Slowest call on each tick of the slowest game, in machine units."""
    points = [(t, s / unit, a) for t, s, a in timeline["points"]]
    if len(points) < 2:
        return ""
    W, H, left, right, top, bottom = 800, 240, 56, 16, 16, 30
    peak = (timeline["peak"][0], timeline["peak"][1] / unit, timeline["peak"][2])
    budget = timeline["budget_s"] / unit
    budget_text = fmt_units(budget) + (" unit" if budget == 1 else " units")
    show_budget = budget <= peak[1] * 4
    y_max = _nice_ceiling(max(max(p[1] for p in points), budget if show_budget else 0) * 1.1)
    t0, t1 = points[0][0], points[-1][0]

    def x(tick: int) -> float:
        return left + (tick - t0) / (t1 - t0) * (W - left - right)

    def y(sec: float) -> float:
        return top + (1 - sec / y_max) * (H - top - bottom)

    base = y(0)
    path = " ".join(f"{'M' if i == 0 else 'L'}{x(t):.1f},{y(s):.1f}" for i, (t, s, _) in enumerate(points))
    area = f"M{x(t0):.1f},{base:.1f} " + path.replace("M", "L", 1) + f" L{x(t1):.1f},{base:.1f} Z"
    svg = [f'<svg id="timeline" viewBox="0 0 {W} {H}" role="img" aria-label="Slowest call per tick" data-points="{escape(json.dumps([{"x": round(x(t), 1), "y": round(y(s), 1), "tick": t, "t": fmt_units(s) + " units", "agent": a} for t, s, a in points]))}">']
    for i in range(5):
        value = y_max * i / 4
        gy = y(value)
        svg.append(f'<line x1="{left}" x2="{W - right}" y1="{gy:.1f}" y2="{gy:.1f}" stroke="var(--{"axis" if i == 0 else "grid"})"/>')
        svg.append(f'<text x="{left - 8}" y="{gy + 4:.1f}" text-anchor="end">{fmt_units(value) if i else "0"}</text>')
    step = max(1, round((t1 - t0) / 8))
    for tick in range(t0, t1 + 1, step):
        svg.append(f'<text x="{x(tick):.1f}" y="{H - 10}" text-anchor="middle">{tick}</text>')
    if show_budget:
        by = y(budget)
        svg.append(f'<line x1="{left}" x2="{W - right}" y1="{by:.1f}" y2="{by:.1f}" stroke="var(--ink-2)"/>')
        svg.append(f'<text x="{W - right}" y="{by - 5:.1f}" text-anchor="end" class="label">budget {budget_text}</text>')
    svg.append(f'<path d="{area}" fill="var(--series)" fill-opacity="0.10"/>')
    svg.append(f'<path d="{path}" fill="none" stroke="var(--series)" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
    px, py = x(peak[0]), y(peak[1])
    anchor, dx = ("end", -8) if px > W * 0.7 else ("start", 8)
    svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="var(--series)" stroke="var(--surface)" stroke-width="2"/>')
    svg.append(f'<text x="{px + dx:.1f}" y="{max(py - 6, 11):.1f}" text-anchor="{anchor}" class="label">{fmt_units(peak[1])} units · {escape(peak[2])}, tick {peak[0]}</text>')
    svg.append(f'<line id="hair" y1="{top}" y2="{base:.1f}" stroke="var(--axis)" style="display:none"/>')
    svg.append('<circle id="dot" r="4" fill="var(--series)" stroke="var(--surface)" stroke-width="2" style="display:none"/>')
    svg.append("</svg>")
    off_chart = "" if show_budget else f" The budget for one call, {budget_text}, is far above this chart."
    return (
        f'<div class="card"><h2>Slowest call on each tick — {escape(timeline["config"])}</h2>'
        f'<p class="sub">The game containing the slowest call of the run, by tick, in machine units. Tick {points[0][0]} holds the first call of each agent.{off_chart}</p>{"".join(svg)}</div>'
    )


def _cell(seconds: float, unit: float, limit: float) -> str:
    return f"{fmt_units(seconds / unit)} <small>{_fmt_pct(seconds / unit / limit)}</small>"


def build_report(r: Dict[str, Any]) -> str:
    unit, rows = r["unit"], r["rows"]
    mark = {"PASS": "✓", "FAIL": "✕", "MEASURED": "–"}[r["verdict"]]
    if r["verdict"] == "PASS":
        summary = "Every size stayed inside its budget."
    elif r["verdict"] == "FAIL":
        summary = "This policy failed the runtime check and would not be accepted."
    else:
        summary = "Measure-only run: timings are reported, no verdict is given."
    reasons = "".join(f"<li>{escape(x)}</li>" for x in r["reasons"]) if r["verdict"] != "PASS" else ""

    calls = sum(row["calls"] for row in rows)
    worst_mean = max(rows, key=lambda row: row["mean_s"], default=None)
    worst_max = max(rows, key=lambda row: row["max_s"], default=None)
    tiles = [
        ("Machine unit", f"{unit:.2f} s", "1 unit on this machine; every time below is in units"),
        ("Games played", str(r["games"]), "one per size (quick)" if r["quick"] else "every example config"),
        ("Calls timed", f"{calls:,}", f"{r['team']} team's strategy calls"),
    ]
    if worst_mean:
        tiles.append(("Highest mean per call (units)", fmt_units(worst_mean["mean_s"] / unit), worst_mean["name"]))
        tiles.append(("Slowest call (units)", fmt_units(worst_max["max_s"] / unit), worst_max["max_where"]))
    payoff = r.get("payoff")
    if payoff:
        sign = 1 if r["team"] == "red" else -1  # the components are red's; blue gets their negatives
        tiles.append((
            f"Mean {r['team']} payoff",
            f"{payoff['team']:+.2f}",
            f"per game vs the opponent · capture {sign * payoff['capture']:+.2f}, tag {sign * payoff['tag']:+.2f}, discover {sign * payoff['discover']:+.2f}",
        ))
    tile_html = "".join(
        f'<div class="tile"><div class="label">{escape(a)}</div><div class="value">{escape(b)}</div><div class="note">{escape(c)}</div></div>'
        for a, b, c in tiles
    )

    panels = "".join(
        _bar_panel(title, sub, rows, key, budget_key, unit)
        for title, sub, key, budget_key in (
            ("Mean time per call", "Share of budget, by size", "mean_s", "mean_per_call"),
            ("Slowest call", "Share of budget, by size", "max_s", "max_per_call"),
            ("First call", "Each agent's first call; share of budget", "first_s", "first_call"),
        )
    ) if rows else ""

    table_rows = "".join(
        f"<tr><td>{escape(row['name'])}</td><td>{row['games']}</td><td>{row['calls']:,}</td>"
        f"<td>{_cell(row['mean_s'], unit, row['budget']['mean_per_call'])}</td>"
        f"<td>{_cell(row['max_s'], unit, row['budget']['max_per_call'])}</td>"
        f"<td>{_cell(row['first_s'], unit, row['budget']['first_call'])}</td>"
        f"<td>{'' if row.get('payoff') is None else format(row['payoff'], '+.2f')}</td>"
        f"<td>{escape(row['max_where'])}</td>"
        f"<td>{'✓ pass' if not row['fails'] else '✕ ' + escape('; '.join(row['fails']))}</td></tr>"
        for row in rows
    )
    notes = [f"Growth: {x}" for x in r["growth"]] + [f"Warning: {x}" for x in r["warnings"]]
    notes_html = (
        '<div class="card"><h2>Notes</h2><ul class="notes">' + "".join(f"<li>{escape(x)}</li>" for x in notes) + "</ul></div>"
        if notes else ""
    )
    drift = abs(r["unit_end"] - unit) / unit * 100

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Policy runtime check — {escape(r['policy'])}</title><style>{_CSS}</style></head>
<body><main>
<h1>Policy runtime check</h1>
<p class="sub">{escape(r['policy'])} · {escape(r['team'])} team · against {escape(r['opponent'])}</p>
<p class="meta">{escape(r['when'])} · Python {escape(r['python'])} · {escape(r['machine'])} · machine unit {unit:.3f} s at start, {r['unit_end']:.3f} s at end ({drift:.0f}% apart)</p>
<div class="card verdict {r['verdict']}"><div class="badge"><span class="mark">{mark}</span> {r['verdict']}</div>
<div><p class="sub">{summary}</p><ul>{reasons}</ul></div></div>
<div class="tiles">{tile_html}</div>
<div class="panels">{panels}</div>
{_timeline(r['timeline'], unit) if r['timeline'] else ''}
<div class="card"><h2>By size</h2>
<p class="sub">Times are in machine units, each followed by its share of the budget. Payoff is the policy team's mean per game. "Mean" and "slowest" leave out each agent's first call, which is budgeted on its own.</p>
<div class="scroll"><table><thead><tr><th>Size</th><th>Games</th><th>Calls</th><th>Mean per call</th><th>Slowest call</th><th>First call</th><th>Payoff</th><th>Slowest call at</th><th>Result</th></tr></thead>
<tbody>{table_rows}</tbody></table></div></div>
{notes_html}
<p class="meta">All times are in machine units — multiples of the time this machine needs for a fixed reference workload (here {unit:.3f} s) — so the same numbers and limits apply on any machine. Hover a bar for the time in seconds. Accuracy is tens of percent, which is enough to catch a policy that is many times too slow.</p>
</main><div class="tip" id="tip"></div><script>{_JS}</script></body></html>
"""


def write_report(result: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_report(result), encoding="utf-8")
