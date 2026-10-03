"""
Turn run logs into report-ready results.

    python analyze.py runs/metrics_baseline.csv
    python analyze.py runs/metrics_baseline.csv runs/metrics_drop30.csv runs/metrics_jit15.csv

Writes to runs/report/:
    summary.md      results table (paste into the report)
    summary.json    the same numbers, machine-readable
    fig_tracking_<tag>.png   error over time, shaded by tracking mode
    fig_safety_<tag>.png     distance to nearest critical structure + thresholds
    fig_latency.png          inference latency distribution, all runs
    fig_comparison.png       headline metrics side by side (2+ runs)

Every number is computed from the CSV the run wrote, so the report can be
regenerated at any time and nothing is typed in by hand.
"""
import argparse
import csv
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join("runs", "report")
MODE_COLOUR = {"track": "#dff3e4", "reacquire": "#fff1cc", "lost": "#fbdada",
               "hold": "#e3ecfa"}


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} is empty - did the run log any frames?")

    def col(name, fn=float, default=np.nan):
        out = []
        for r in rows:
            v = r.get(name, "")
            try:
                out.append(fn(v) if v != "" else default)
            except ValueError:
                out.append(default)
        return np.array(out, dtype=object if fn is str else float)

    return {
        "t": col("t"), "frame": col("frame"), "infer": col("infer_ms"),
        "mode": col("track_mode", str, ""), "in_view": col("in_view"),
        "err": col("err_norm"), "dist": col("dist_px"),
        "warn": col("warn_px"), "danger": col("danger_px"),
        "level": col("level", str, ""), "pred": col("predictive", float, 0),
        "ttc": col("ttc_s"), "clear": col("arm_clear_m"),
        "guard": col("arm_guard", float, 0),
        "pivot": col("cam_pivot_err_mm"), "cmd": col("last_command", str, ""),
    }


def _entries(mask):
    """Count transitions into a True state (events, not frames)."""
    m = np.asarray(mask, bool)
    return int(np.sum(m[1:] & ~m[:-1]) + (1 if m.size and m[0] else 0))


def summarise(d, tag):
    n = len(d["t"])
    dur = float(d["t"][-1] - d["t"][0]) if n > 1 else 0.0
    iv = d["in_view"] == 1
    track = d["mode"] == "track"
    err_iv = d["err"][iv]
    inf = d["infer"][~np.isnan(d["infer"])]
    dist = d["dist"][~np.isnan(d["dist"])]
    ttc = d["ttc"][~np.isnan(d["ttc"])]
    clear = d["clear"][~np.isnan(d["clear"])]
    frames_unique = len(np.unique(d["frame"][~np.isnan(d["frame"])]))
    cmds = [c for c in d["cmd"] if c]
    n_cmds = sum(1 for i in range(1, len(cmds)) if cmds[i] != cmds[i - 1]) \
        + (1 if cmds else 0)

    def pct(x):
        return float(100.0 * np.mean(x)) if len(x) else float("nan")

    def q(x, p):
        return float(np.percentile(x, p)) if len(x) else float("nan")

    return {
        "tag": tag,
        "control_steps": n,
        "video_frames": frames_unique,
        "duration_s": round(dur, 1),
        "in_view_pct": round(pct(iv), 1),
        "track_mode_pct": round(pct(track), 1),
        "err_mean": round(float(np.mean(err_iv)), 4) if err_iv.size else None,
        "err_median": round(q(err_iv, 50), 4) if err_iv.size else None,
        "err_p95": round(q(err_iv, 95), 4) if err_iv.size else None,
        "infer_mean_ms": round(float(np.mean(inf)), 1) if inf.size else None,
        "infer_p95_ms": round(q(inf, 95), 1) if inf.size else None,
        "warn_pct": round(pct(d["level"] == "warn"), 1),
        "danger_pct": round(pct(d["level"] == "danger"), 1),
        "danger_events": _entries(d["level"] == "danger"),
        "predictive_events": _entries(d["pred"] == 1),
        "min_dist_px": round(float(np.min(dist)), 1) if dist.size else None,
        "min_ttc_s": round(float(np.min(ttc)), 2) if ttc.size else None,
        "min_arm_clearance_cm": round(100 * float(np.min(clear)), 1)
        if clear.size else None,
        "arm_guard_events": _entries(d["guard"] == 1),
        "max_pivot_err_mm": round(float(np.nanmax(d["pivot"])), 6)
        if np.any(~np.isnan(d["pivot"])) else None,
        "commands": n_cmds,
    }


def fig_tracking(d, tag):
    fig, ax = plt.subplots(figsize=(10, 3.2))
    t = d["t"] - d["t"][0]
    # shade tracking modes
    start = 0
    for i in range(1, len(t) + 1):
        if i == len(t) or d["mode"][i] != d["mode"][start]:
            c = MODE_COLOUR.get(d["mode"][start])
            if c:
                ax.axvspan(t[start], t[min(i, len(t) - 1)], color=c, lw=0)
            start = i
    err = np.where(d["in_view"] == 1, d["err"], np.nan)
    ax.plot(t, err, lw=1.0, color="#2a6f97", label="centroid error (in view)")
    ax.axhline(0.06, ls="--", lw=0.8, color="#888", label="deadband")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("normalised error")
    ax.set_ylim(0, max(0.5, float(np.nanmax(err)) * 1.1
                       if np.any(~np.isnan(err)) else 1.0))
    ax.set_title(f"Camera tracking - {tag}  (green=track, yellow=reacquire, "
                 f"red=lost, blue=hold)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    p = os.path.join(OUT, f"fig_tracking_{tag}.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_safety(d, tag):
    if not np.any(~np.isnan(d["dist"])):
        return None
    fig, ax = plt.subplots(figsize=(10, 3.2))
    t = d["t"] - d["t"][0]
    ax.plot(t, d["dist"], lw=1.0, color="#333", label="tip to nearest critical")
    ax.plot(t, d["warn"], ls="--", lw=0.9, color="#e0a100", label="warn")
    ax.plot(t, d["danger"], ls="--", lw=0.9, color="#d62828", label="danger")
    pred = d["pred"] == 1
    if np.any(pred):
        ax.scatter(t[pred], d["dist"][pred], s=8, color="#e0a100", zorder=3,
                   label="predictive (TTC) warning")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("distance (px)")
    ax.set_title(f"Safety margin - {tag}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    p = os.path.join(OUT, f"fig_safety_{tag}.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_latency(runs):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for tag, d in runs:
        x = d["infer"][~np.isnan(d["infer"])]
        if x.size:
            ax.hist(x, bins=40, alpha=0.55, label=tag)
    ax.set_xlabel("inference latency (ms)")
    ax.set_ylabel("control steps")
    ax.set_title("Perception latency")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_latency.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_comparison(summaries):
    tags = [s["tag"] for s in summaries]
    iv = [s["in_view_pct"] for s in summaries]
    em = [s["err_mean"] or 0 for s in summaries]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4))
    a.bar(tags, iv, color="#2a9d8f")
    a.set_ylabel("instrument in view (%)")
    a.set_ylim(0, 105)
    for i, v in enumerate(iv):
        a.text(i, v + 1.5, f"{v:.1f}", ha="center", fontsize=8)
    b.bar(tags, em, color="#e76f51")
    b.set_ylabel("mean centroid error")
    for i, v in enumerate(em):
        b.text(i, v * 1.02, f"{v:.3f}", ha="center", fontsize=8)
    for ax in (a, b):
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Tracking robustness across runs")
    fig.tight_layout()
    p = os.path.join(OUT, "fig_comparison.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def write_md(summaries, figs):
    rows = [
        ("Duration (s)", "duration_s"),
        ("Video frames processed", "video_frames"),
        ("Instrument in view (%)", "in_view_pct"),
        ("Time in TRACK mode (%)", "track_mode_pct"),
        ("Centroid error, mean", "err_mean"),
        ("Centroid error, median", "err_median"),
        ("Centroid error, 95th pct", "err_p95"),
        ("Inference latency, mean (ms)", "infer_mean_ms"),
        ("Inference latency, 95th pct (ms)", "infer_p95_ms"),
        ("Time in WARN (%)", "warn_pct"),
        ("Time in DANGER (%)", "danger_pct"),
        ("Danger events", "danger_events"),
        ("Predictive (TTC) warnings", "predictive_events"),
        ("Min distance to critical (px)", "min_dist_px"),
        ("Min time-to-contact (s)", "min_ttc_s"),
        ("Min arm-to-arm clearance (cm)", "min_arm_clearance_cm"),
        ("Arm-guard activations", "arm_guard_events"),
        ("Max RCM pivot error (mm)", "max_pivot_err_mm"),
        ("Commands issued", "commands"),
    ]
    head = "| Metric | " + " | ".join(s["tag"] for s in summaries) + " |"
    sep = "|---|" + "---|" * len(summaries)
    lines = ["# Results", "", head, sep]
    for label, key in rows:
        vals = ["-" if s.get(key) is None else str(s[key]) for s in summaries]
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    lines += ["", "## Figures", ""]
    lines += [f"![{os.path.basename(f)}]({os.path.basename(f)})"
              for f in figs if f]
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="*", default=["runs/metrics_run.csv"])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    runs, summaries, figs = [], [], []
    for path in args.csv:
        tag = os.path.splitext(os.path.basename(path))[0].replace("metrics_", "")
        d = load(path)
        runs.append((tag, d))
        summaries.append(summarise(d, tag))
        figs.append(fig_tracking(d, tag))
        figs.append(fig_safety(d, tag))
    figs.append(fig_latency(runs))
    if len(runs) > 1:
        figs.append(fig_comparison(summaries))

    write_md(summaries, figs)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summaries, f, indent=2)

    for s in summaries:
        print(f"\n[{s['tag']}]")
        for k, v in s.items():
            if k != "tag":
                print(f"  {k:24s} {v}")
    print(f"\nreport written to {OUT}/summary.md")


if __name__ == "__main__":
    main()
