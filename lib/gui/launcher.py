import glob
import os
import pickle
import threading
import tkinter as tk
import yaml
from pathlib import Path
from tkinter import filedialog

import matplotlib.figure
import matplotlib.patches as patches
import networkx as nx
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.collections import LineCollection

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_DEFAULTS = {
    "config_main": "example/example_config.yml",
    "extra_defs": "config/game_config.yml",
    "red_strategy": "example_atk",
    "blue_strategy": "example_def",
    "log_level": "DEBUG",
}

PREVIEW_PX = 600


# ── Discovery ─────────────────────────────────────────────────────────────────

def _discover_configs() -> list[str]:
    example_ymls = sorted(glob.glob(str(_REPO_ROOT / "example" / "*.yml")))
    config_ymls  = sorted(glob.glob(str(_REPO_ROOT / "config" / "**" / "*.yml"), recursive=True))
    return [
        os.path.relpath(p, _REPO_ROOT)
        for p in example_ymls + config_ymls
        if not os.path.basename(p).startswith("game_config")
        and "archive" not in Path(p).relative_to(_REPO_ROOT).parts
        and "rules" not in Path(p).relative_to(_REPO_ROOT).parts
    ] or ["(none found)"]


def _discover_extra_defs() -> list[str]:
    paths = sorted(glob.glob(str(_REPO_ROOT / "config" / "game_config*.yml")))
    return [os.path.relpath(p, _REPO_ROOT) for p in paths] or ["config/game_config.yml"]


def _discover_strategies(subdir: str, prefix: str, example: str) -> tuple[list[str], dict[str, str]]:
    stems = sorted(
        Path(p).stem
        for p in glob.glob(str(_REPO_ROOT / subdir / "*.py"))
        if not Path(p).stem.startswith("__")
    )
    example_stem = example.split(".")[-1]
    names   = [example_stem] + stems
    mapping = {example_stem: example, **{s: f"{prefix}.{s}" for s in stems}}
    return names, mapping


def _pick(options: list[str], default: str) -> str:
    return default if default in options else (options[0] if options else default)


# ── Config info ───────────────────────────────────────────────────────────────

def _load_config_info(rel_path: str) -> dict:
    try:
        with open(_REPO_ROOT / rel_path) as f:
            cfg = yaml.safe_load(f)
        red_cfg  = (cfg.get("agents") or {}).get("red_config")  or {}
        blue_cfg = (cfg.get("agents") or {}).get("blue_config") or {}
        flags    = cfg.get("flags") or {}
        return {
            "graph":      (cfg.get("environment") or {}).get("graph_name", "—"),
            "red":        len(red_cfg),
            "blue":       len(blue_cfg),
            "flags":      len(flags.get("real_positions") or []),
            "candidates": len(flags.get("candidate_positions") or []),
            "max_time":   (cfg.get("game") or {}).get("max_time", "—"),
        }
    except Exception:
        return {}


# ── Graph rendering ───────────────────────────────────────────────────────────

def _load_graph(graph_name: str):
    for root in [_REPO_ROOT / "graphs", _REPO_ROOT]:
        p = root / graph_name
        if p.exists():
            with open(p, "rb") as f:
                return pickle.load(f)
    raise FileNotFoundError(f"Graph not found: {graph_name}")


def _node_pos(G, node_id) -> tuple:
    d = G.nodes[node_id]
    if "x" in d and "y" in d:
        return d["x"], d["y"]
    if "pos" in d:
        return tuple(d["pos"])[:2]
    w = 200
    return node_id % w, node_id // w


def _render_to_fig(fig: matplotlib.figure.Figure, rel_path: str) -> bool:
    try:
        with open(_REPO_ROOT / rel_path) as f:
            cfg = yaml.safe_load(f)

        G   = _load_graph(cfg["environment"]["graph_name"])
        pos = {n: _node_pos(G, n) for n in G.nodes()}

        ax = fig.add_subplot(111)
        ax.set_aspect("equal")
        ax.axis("off")

        # Edges
        lines = []
        for u, v, data in G.edges(data=True):
            geom = data.get("linestring")
            if geom is not None and hasattr(geom, "coords"):
                lines.append(list(geom.coords))
            elif u in pos and v in pos:
                lines.append([pos[u], pos[v]])
        if lines:
            ax.add_collection(LineCollection(lines, colors="#555555", alpha=0.3, linewidths=0.8))

        # Graph nodes
        gxs = [pos[n][0] for n in G.nodes() if n in pos]
        gys = [pos[n][1] for n in G.nodes() if n in pos]
        ax.scatter(gxs, gys, s=4, c="#333333", alpha=0.4, zorder=2)

        # Sensing radii
        def draw_radius(node_id, r, color):
            if node_id not in pos or not r:
                return
            xy = pos[node_id]
            ax.add_patch(patches.Circle(xy, r, color=color, alpha=0.08, linewidth=0))
            ax.add_patch(patches.Circle(xy, r, color=color, fill=False, alpha=0.3,
                                        linewidth=0.8, linestyle="--"))

        env    = cfg.get("environment", {})
        agents = cfg.get("agents", {})

        stat_r = env.get("blue_stationary_sensor_radius", 0)
        for n in env.get("blue_static_sensor_positions") or []:
            draw_radius(n, stat_r, "#00cc00")

        red_r = (agents.get("red_global") or {}).get("sensing_radius", 0)
        for d in (agents.get("red_config") or {}).values():
            draw_radius(d["start_node_id"], red_r, "red")

        blue_r = (agents.get("blue_global") or {}).get("sensing_radius", 0)
        for d in (agents.get("blue_config") or {}).values():
            draw_radius(d["start_node_id"], blue_r, "blue")

        # Flags
        real_flags = set(cfg.get("flags", {}).get("real_positions") or [])
        cands      = set(cfg.get("flags", {}).get("candidate_positions") or [])
        fakes      = cands - real_flags

        def scatter_nodes(node_ids, **kw):
            pts = [pos[n] for n in node_ids if n in pos]
            if pts:
                ax.scatter([p[0] for p in pts], [p[1] for p in pts], **kw)

        scatter_nodes(fakes,      s=60, c="gray",    marker="s", zorder=4, edgecolors="black", linewidths=0.5)
        scatter_nodes(real_flags, s=60, c="#00cc00",  marker="s", zorder=5, edgecolors="black", linewidths=0.5)

        red_nodes  = [d["start_node_id"] for d in (agents.get("red_config")  or {}).values()]
        blue_nodes = [d["start_node_id"] for d in (agents.get("blue_config") or {}).values()]
        scatter_nodes(red_nodes,  s=80, c="red",  zorder=6, edgecolors="black", linewidths=0.5)
        scatter_nodes(blue_nodes, s=80, c="blue", zorder=6, edgecolors="white", linewidths=0.5)

        ax.autoscale_view()
        fig.tight_layout(pad=0.3)
        return True
    except Exception:
        return False


# ── File dialogs ──────────────────────────────────────────────────────────────

def _browse_yaml(var: tk.StringVar, on_change=None) -> None:
    path = filedialog.askopenfilename(
        title="Select config file",
        initialdir=str(_REPO_ROOT / "config"),
        filetypes=[("YAML files", "*.yml *.yaml"), ("All files", "*")],
    )
    if path:
        try:
            rel = os.path.relpath(path, _REPO_ROOT)
        except ValueError:
            rel = path
        var.set(rel)
        if on_change:
            on_change(rel)


def _browse_tiff(var: tk.StringVar) -> None:
    path = filedialog.askopenfilename(
        title="Select TIFF file",
        initialdir=str(_REPO_ROOT / "graphs"),
        filetypes=[("TIFF files", "*.tif *.tiff *.TIFF"), ("All files", "*")],
    )
    if path:
        try:
            rel = os.path.relpath(path, _REPO_ROOT)
        except ValueError:
            rel = path
        var.set(rel)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    os.chdir(_REPO_ROOT)

    configs              = _discover_configs()
    extra_defs           = _discover_extra_defs()
    red_strats, red_map  = _discover_strategies("policies/attacker", "policies.attacker", "example.example_atk")
    blue_strats, blue_map = _discover_strategies("policies/defender", "policies.defender", "example.example_def")
    levels               = ["DEBUG", "INFO", "WARNING", "ERROR"]

    root = tk.Tk()
    root.title("League Runner")
    root.resizable(False, False)

    # Scale figure DPI to match display density (retina = 2×, standard = 1×)
    try:
        dpi_scale = float(root.tk.call("tk", "scaling"))
    except Exception:
        dpi_scale = 1.0
    dpi_scale = max(1.0, min(dpi_scale, 3.0))
    fig_dpi   = int(100 * dpi_scale)

    # ── Variables ─────────────────────────────────────────────────────────────
    config_var    = tk.StringVar(value=_pick(configs,    _DEFAULTS.get("config_main", "")))
    extra_var     = tk.StringVar(value=_pick(extra_defs, _DEFAULTS["extra_defs"]))
    red_var       = tk.StringVar(value=_pick(red_strats, _DEFAULTS["red_strategy"]))
    blue_var      = tk.StringVar(value=_pick(blue_strats,_DEFAULTS["blue_strategy"]))
    log_name_var  = tk.StringVar(value="")
    level_var     = tk.StringVar(value=_DEFAULTS["log_level"])
    rec_file_var  = tk.BooleanVar(value=False)
    rec_video_var = tk.BooleanVar(value=False)
    vis_var       = tk.BooleanVar(value=True)
    tiff_var      = tk.StringVar(value="")

    FONT = ("TkDefaultFont", 15)

    # ── Root layout: [left controls] | [right preview] ────────────────────────
    outer = tk.Frame(root, padx=14, pady=12)
    outer.pack()

    left = tk.Frame(outer)
    left.grid(row=0, column=0, sticky="n")

    tk.Frame(outer, width=1, bg="#bbbbbb").grid(row=0, column=1, sticky="ns", padx=12)

    right = tk.Frame(outer)
    right.grid(row=0, column=2, sticky="n")

    # ── Right: pure matplotlib canvas ─────────────────────────────────────────
    fig    = matplotlib.figure.Figure(figsize=(PREVIEW_PX / 100, PREVIEW_PX / 100), dpi=fig_dpi)
    canvas = FigureCanvasTkAgg(fig, master=right)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.config(width=PREVIEW_PX, height=PREVIEW_PX, highlightthickness=0)
    canvas_widget.pack()

    # ── Preview update (background thread + generation counter) ────────────────
    _gen  = [0]
    _lock = threading.Lock()

    # Info labels live on the left (populated by update_preview below)
    graph_lbl  = tk.Label(left, text="Graph:     —", anchor="w", font=FONT)
    counts_lbl = tk.Label(left, text="Red: —   Blue: —", anchor="w", font=FONT)
    flags_lbl  = tk.Label(left, text="Flags: —   Candidates: —", anchor="w", font=FONT)
    time_lbl   = tk.Label(left, text="Max Time: —", anchor="w", font=FONT)

    def update_preview(rel_path: str) -> None:
        info = _load_config_info(rel_path)
        graph_lbl .config(text=f"Graph:     {info.get('graph', '—')}")
        counts_lbl.config(text=f"Red: {info.get('red', '—')}   Blue: {info.get('blue', '—')}")
        flags_lbl .config(text=f"Flags: {info.get('flags', '—')}   Candidates: {info.get('candidates', '—')}")
        time_lbl  .config(text=f"Max Time: {info.get('max_time', '—')}")

        _gen[0] += 1
        gen = _gen[0]

        def _render():
            with _lock:
                if _gen[0] != gen:
                    return
                fig.clear()
                ok = _render_to_fig(fig, rel_path)
                if not ok:
                    ax = fig.add_subplot(111)
                    ax.axis("off")
                    ax.text(0.5, 0.5, "No preview available", ha="center", va="center",
                            transform=ax.transAxes, color="#888888", fontsize=13)
            if _gen[0] == gen:
                root.after(0, canvas.draw)

        threading.Thread(target=_render, daemon=True).start()

    config_var.trace_add("write", lambda *_: update_preview(config_var.get()))

    # ── Left: helpers ──────────────────────────────────────────────────────────
    menu_w = 38

    def lbl(text: str, row: int) -> None:
        tk.Label(left, text=text, anchor="w", font=FONT).grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=3
        )

    def opt(parent, var, options, row, col=1, colspan=2) -> tk.OptionMenu:
        m = tk.OptionMenu(parent, var, *options)
        m.config(font=FONT, width=menu_w, anchor="w")
        m["menu"].config(font=FONT)
        m.grid(row=row, column=col, columnspan=colspan, sticky="ew", pady=3)
        return m

    def sep(row: int) -> None:
        tk.Frame(left, height=1, bg="#cccccc").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=(6, 2)
        )

    # ── Left: controls ─────────────────────────────────────────────────────────
    lbl("Config:", 0)
    opt(left, config_var, configs, row=0)
    tk.Button(left, text="Browse…", font=FONT,
              command=lambda: _browse_yaml(config_var, update_preview)).grid(
        row=0, column=2, padx=(6, 0), pady=3
    )

    lbl("Extra defs:", 1)
    opt(left, extra_var, extra_defs, row=1)

    sep(2)

    lbl("Red strategy:", 3)
    opt(left, red_var, red_strats, row=3)

    lbl("Blue strategy:", 4)
    opt(left, blue_var, blue_strats, row=4)

    sep(5)

    lbl("Log name:", 6)
    tk.Entry(left, textvariable=log_name_var, font=FONT, width=menu_w + 2).grid(
        row=6, column=1, columnspan=2, sticky="ew", pady=3
    )

    lbl("Log level:", 7)
    opt(left, level_var, levels, row=7)

    sep(8)

    tk.Checkbutton(left, text="Record file (.ggr)",  variable=rec_file_var,  font=FONT, anchor="w").grid(row=9,  column=0, columnspan=3, sticky="w", pady=2)
    tk.Checkbutton(left, text="Record video (.mp4)", variable=rec_video_var, font=FONT, anchor="w").grid(row=10, column=0, columnspan=3, sticky="w", pady=2)
    tk.Checkbutton(left, text="Visualization",        variable=vis_var,       font=FONT, anchor="w").grid(row=11, column=0, columnspan=3, sticky="w", pady=2)

    sep(12)

    lbl("TIFF path:", 13)
    tk.Entry(left, textvariable=tiff_var, font=FONT, width=menu_w + 2).grid(
        row=13, column=1, sticky="ew", pady=3
    )
    tk.Button(left, text="Browse…", font=FONT, command=lambda: _browse_tiff(tiff_var)).grid(
        row=13, column=2, padx=(6, 0), pady=3
    )

    sep(14)

    graph_lbl .grid(row=15, column=0, columnspan=3, sticky="w", pady=1)
    counts_lbl.grid(row=16, column=0, columnspan=3, sticky="w", pady=1)
    flags_lbl .grid(row=17, column=0, columnspan=3, sticky="w", pady=1)
    time_lbl  .grid(row=18, column=0, columnspan=3, sticky="w", pady=1)

    # Trigger initial preview
    update_preview(config_var.get())

    # ── Center on screen ───────────────────────────────────────────────────────
    root.update_idletasks()
    w, h = root.winfo_reqwidth(), root.winfo_reqheight()
    x = (root.winfo_screenwidth()  - w) // 2
    y = (root.winfo_screenheight() - h) // 2
    root.geometry(f"+{x}+{y}")

    # ── Run button ─────────────────────────────────────────────────────────────
    def on_run() -> None:
        cfg        = config_var.get()
        extra      = extra_var.get()
        red        = red_map[red_var.get()]
        blue       = blue_map[blue_var.get()]
        log_name   = log_name_var.get().strip() or None
        level_name = level_var.get()
        rec_file   = rec_file_var.get()
        rec_video  = rec_video_var.get()
        vis        = vis_var.get()
        tiff       = tiff_var.get().strip() or None
        root.destroy()

        from lib.core.console import LogLevel
        from lib.game.game_engine import GameEngine

        GameEngine.launch_from_files(
            config_main=cfg,
            extra_defs=extra,
            red_strategy=red,
            blue_strategy=blue,
            log_name=log_name,
            set_level=getattr(LogLevel, level_name),
            record_file=rec_file,
            record_video=rec_video,
            vis=vis,
            tiff_path=tiff,
        )

    tk.Button(
        root,
        text="RUN GAME",
        command=on_run,
        font=("TkDefaultFont", 15, "bold"),
        padx=20,
        pady=8,
    ).pack(pady=(4, 14))

    root.mainloop()
