"""
Reads JSON results from simulations, summarizes metrics (mean, std, CI),
and produces publication-ready plots.

- If ndrones has multiple values: line plot vs #UAVs with 95% CI band
- If ndrones has a single value: bar plot comparing algorithms with 95% CI error bars
- Adds tail-delay metrics computed from per-event delivery times:
    * event_p95_delivery_time
    * event_max_delivery_time
- Y-axis:
    * For ratio metrics (and routing ratio): fixed to (0, 1)
    * For other metrics: automatically zoom to (mean ± 95% CI) range with margin
- Saves figures as PNG (dpi=400), PDF, SVG into data/plots/
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from argparse import ArgumentParser
from src.utilities import config

# =========================================================
# Global setup
# =========================================================
DEFAULT_OUT_DIR = "data/plots"
os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)

mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "legend.fontsize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "lines.linewidth": 2.2,
    "lines.markersize": 7,
    "axes.grid": True,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.6,
})

# =========================================================
# Metrics
# =========================================================
METRICS_OF_INTEREST = [
    "number_of_generated_events",
    "number_of_detected_events",
    "number_of_events_to_depot",
    "number_of_packets_to_depot",
    "packet_mean_delivery_time",
    "event_mean_delivery_time",

    # ✅ Tail metrics (NEW)
    "event_p95_delivery_time",
    "event_max_delivery_time",

    "time_on_mission",
    "time_on_active_routing",
    "Routing time / mission time",
    "ratio_delivery_generated",
    "ratio_delivery_detected",
]

METRIC_LABEL = {
    "number_of_generated_events": "Generated events (#)",
    "number_of_detected_events": "Detected events (#)",
    "number_of_events_to_depot": "Delivered events to depot (#)",
    "number_of_packets_to_depot": "Delivered packets to depot (#)",
    "packet_mean_delivery_time": "Mean packet delivery time (s)",
    "event_mean_delivery_time": "Mean event delivery time (s)",

    # ✅ Tail labels (NEW)
    "event_p95_delivery_time": "Event delivery time p95 (s)",
    "event_max_delivery_time": "Event delivery time max (s)",

    "time_on_mission": "Time on mission (steps)",
    "time_on_active_routing": "Time on active routing (steps)",
    "Routing time / mission time": "Active routing / mission time (ratio)",
    "ratio_delivery_generated": "Delivery ratio (delivered / generated)",
    "ratio_delivery_detected": "Delivery ratio (delivered / detected)",
}

# Metrics where it is better/safer to keep fixed y-limits
METRIC_YLIM = {
    "ratio_delivery_generated": (0.0, 1.0),
    "ratio_delivery_detected": (0.0, 1.0),
    "Routing time / mission time": (0.0, 1.0),
}

AUTO_Y_MARGIN_RATIO = 0.15

# =========================================================
# Helper functions
# =========================================================
def _safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)

def _metric_to_filename(metric: str) -> str:
    if metric == "Routing time / mission time":
        return "routing_time_mission_time"
    return metric

def _save_figure(basepath_no_ext: str):
    _safe_makedirs(os.path.dirname(basepath_no_ext))
    plt.savefig(basepath_no_ext + ".png", dpi=400, bbox_inches="tight")
    plt.savefig(basepath_no_ext + ".pdf", bbox_inches="tight")
    plt.savefig(basepath_no_ext + ".svg", bbox_inches="tight")

def _read_json(file_name: str):
    with open(file_name, "r", encoding="utf-8") as fp:
        return json.load(fp)

def _to_float_or_nan(v):
    """Convert numbers/number-strings to float, else NaN."""
    if v is None:
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except Exception:
            return np.nan
    return np.nan

def _extract_event_delivery_times_seconds(js: dict):
    """
    Extract per-event delivery times from JSON and convert to float seconds.
    Expected key: 'events_delivery_times' (list of numbers or numeric strings).
    Returns: np.array([...]) or empty array if missing.
    """
    arr = js.get("events_delivery_times", None)
    if arr is None:
        return np.array([], dtype=float)

    vals = []
    for x in arr:
        fx = _to_float_or_nan(x)
        if not np.isnan(fx):
            vals.append(fx)
    return np.array(vals, dtype=float)

def _compute_metric(js: dict, metric: str):
    # ----- Ratios derived from counts -----
    if metric == "ratio_delivery_generated":
        g = _to_float_or_nan(js.get("number_of_generated_events", 0))
        d = _to_float_or_nan(js.get("number_of_events_to_depot", 0))
        return np.nan if (np.isnan(g) or g == 0) else d / g

    if metric == "ratio_delivery_detected":
        g = _to_float_or_nan(js.get("number_of_detected_events", 0))
        d = _to_float_or_nan(js.get("number_of_events_to_depot", 0))
        return np.nan if (np.isnan(g) or g == 0) else d / g

    # ----- Routing / mission ratio -----
    if metric == "Routing time / mission time":
        tr = _to_float_or_nan(js.get("time_on_active_routing", None))
        tm = _to_float_or_nan(js.get("time_on_mission", None))
        return np.nan if (np.isnan(tr) or np.isnan(tm) or tm == 0) else tr / tm

    # ----- Tail metrics from per-event delivery-time distribution (NEW) -----
    if metric == "event_p95_delivery_time":
        t = _extract_event_delivery_times_seconds(js)
        return np.nan if t.size == 0 else float(np.percentile(t, 95))

    if metric == "event_max_delivery_time":
        t = _extract_event_delivery_times_seconds(js)
        return np.nan if t.size == 0 else float(np.max(t))

    # ----- Standard metrics -----
    return _to_float_or_nan(js.get(metric, np.nan))

def set_auto_ylimit_from_data(ax, means, errs, margin_ratio=AUTO_Y_MARGIN_RATIO):
    """
    Automatically zoom y-axis based on (mean ± err).
    NaNs are ignored.
    """
    means = np.asarray(means, dtype=float)
    errs = np.asarray(errs, dtype=float)

    mask = ~np.isnan(means)
    if not np.any(mask):
        return

    errs2 = np.where(np.isnan(errs), 0.0, errs)

    lows = means[mask] - errs2[mask]
    highs = means[mask] + errs2[mask]

    y_min = float(np.min(lows))
    y_max = float(np.max(highs))

    if y_min == y_max:
        span = 0.5 if y_min == 0 else abs(y_min) * 0.05
        if span == 0:
            span = 0.5
        y_min -= span
        y_max += span
    else:
        margin = margin_ratio * (y_max - y_min)
        y_min -= margin
        y_max += margin

    ax.set_ylim(y_min, y_max)

def apply_dynamic_or_fixed_ylim(ax, metric, means, errs):
    if metric in METRIC_YLIM:
        ax.set_ylim(METRIC_YLIM[metric])
    else:
        set_auto_ylimit_from_data(ax, means, errs, margin_ratio=AUTO_Y_MARGIN_RATIO)

def mean_std_ci95(filename_fmt, nd, alg, seeds, metric):
    data = []
    for s in seeds:
        fn = filename_fmt.format(nd, s, alg)
        if not os.path.exists(fn):
            continue
        v = _compute_metric(_read_json(fn), metric)
        if not np.isnan(v):
            data.append(v)

    if len(data) == 0:
        return np.nan, np.nan, np.nan, 0

    data = np.array(data, dtype=float)
    mean = float(data.mean())
    std = float(data.std(ddof=1)) if len(data) > 1 else 0.0
    ci95 = float(1.96 * std / np.sqrt(len(data))) if len(data) > 1 else 0.0
    return mean, std, ci95, int(len(data))

# =========================================================
# Plot functions
# =========================================================
def plot_fixed_nd_bar(filename_fmt, nd, metric, algs, seeds, out_dir, exp_metric):
    means, ci95s, ns = [], [], []

    for alg in algs:
        m, _, ci, n = mean_std_ci95(filename_fmt, nd, alg, seeds, metric)
        means.append(m)
        ci95s.append(ci)
        ns.append(n)

    if all(n == 0 for n in ns):
        print(
            f"[WARN] No data found for nd={nd}, metric='{metric}', algs={algs}. "
            f"Check filename pattern or exp_metric/suffix."
        )
        return

    x = np.arange(len(algs))
    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    ax.bar(x, means, yerr=ci95s, capsize=5)

    labels = [f"{a}\n(n={n})" for a, n in zip(algs, ns)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_ylabel(METRIC_LABEL.get(metric, metric))
    ax.set_title(f"N={nd} UAVs (mean ± 95% CI)")

    apply_dynamic_or_fixed_ylim(ax, metric, means, ci95s)
    ax.grid(True, axis="y")

    metric_fn = _metric_to_filename(metric)
    base = os.path.join(out_dir, f"_{exp_metric}_", f"fixedN{nd}_{metric_fn}")
    _save_figure(base)
    plt.close(fig)

def plot_vs_nd_line(filename_fmt, nd_list, metric, algs, seeds, out_dir, exp_metric):
    fig, ax = plt.subplots(figsize=(8.8, 6.2))

    all_means = []
    all_cis = []

    for alg in algs:
        means, ci95s = [], []
        for nd in nd_list:
            m, _, ci, _ = mean_std_ci95(filename_fmt, nd, alg, seeds, metric)
            means.append(m)
            ci95s.append(ci)

        means = np.array(means, dtype=float)
        ci95s = np.array(ci95s, dtype=float)

        ax.plot(nd_list, means, marker="o", label=alg)
        ax.fill_between(nd_list, means - ci95s, means + ci95s, alpha=0.18)

        all_means.extend(list(means))
        all_cis.extend(list(ci95s))

    ax.set_xlabel(exp_metric.replace("_", "").upper())
    ax.set_ylabel(METRIC_LABEL.get(metric, metric))
    ax.set_xticks(nd_list)
    ax.legend()

    apply_dynamic_or_fixed_ylim(ax, metric, all_means, all_cis)

    metric_fn = _metric_to_filename(metric)
    base = os.path.join(out_dir, f"_{exp_metric}_", metric_fn)
    _save_figure(base)
    plt.close(fig)

# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-nd", dest="number_of_drones", action="append", type=int, required=True)
    parser.add_argument("-i_s", dest="initial_seed", type=int, required=True)
    parser.add_argument("-e_s", dest="end_seed", type=int, required=True)
    parser.add_argument("-exp_suffix", dest="alg_exp_suffix", action="append", type=str, required=True)
    parser.add_argument("-exp_metric", dest="exp_metric", type=str, default="ndrones_")

    args = parser.parse_args()

    nd_list = sorted(set(args.number_of_drones))
    seeds = list(range(args.initial_seed, args.end_seed))
    algs = args.alg_exp_suffix
    exp_metric = args.exp_metric

    pattern_file = os.path.join(
        config.EXPERIMENTS_DIR,
        f"out__{exp_metric}" + "{}_seed{}_alg_{}.json",
    )

    out_dir = getattr(config, "SAVE_PLOT_DIR", DEFAULT_OUT_DIR)
    _safe_makedirs(os.path.join(out_dir, f"_{exp_metric}_"))

    if len(nd_list) == 1:
        nd = nd_list[0]
        for metric in METRICS_OF_INTEREST:
            plot_fixed_nd_bar(pattern_file, nd, metric, algs, seeds, out_dir, exp_metric)
    else:
        for metric in METRICS_OF_INTEREST:
            plot_vs_nd_line(pattern_file, nd_list, metric, algs, seeds, out_dir, exp_metric)

    print("Plotting completed.")
