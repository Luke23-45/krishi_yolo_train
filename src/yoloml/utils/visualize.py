"""
yoloml/utils/visualize.py
-------------------------
Publication-quality dataset analysis and visualization for YoloML.

Generates presentation-ready figures analyzing class distribution,
imbalance severity, and the theoretical basis for the rebalancing strategy.

Output:
    outputs/figures/01_class_distribution.{png,pdf}
    outputs/figures/02_imbalance_matrix.{png,pdf}
    outputs/figures/03_train_val_split.{png,pdf}
    outputs/figures/04_effective_number.{png,pdf}
    outputs/figures/05_rebalancing_preview.{png,pdf}

References:
    Cui et al., "Class-Balanced Loss Based on Effective Number of Samples", CVPR 2019
    Gupta et al., "LVIS: A Dataset for Large Vocabulary Instance Segmentation", CVPR 2019
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import yaml
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from yoloml.config import VisualizationConfig, setup_config

logger = logging.getLogger("yoloml.visualize")

PROJECT_ROOT = Path(__file__).resolve().parents[3]

BG = "#0D1117"
SURFACE = "#161B22"
BORDER = "#30363D"
TEXT = "#E6EDF3"
TEXT_MUTED = "#8B949E"
GRID = "#21262D"

SEVERITY_STOPS = ["#58A6FF", "#79C0FF", "#D2A8FF", "#FFA657", "#FF7B72"]
SEVERITY_CMAP = LinearSegmentedColormap.from_list("severity", SEVERITY_STOPS)

TRAIN_COLOR = "#58A6FF"
VAL_COLOR = "#F0883E"
ACCENT = "#58A6FF"

BETA_COLORS = {
    0.9: "#8B949E",
    0.99: "#79C0FF",
    0.999: "#D2A8FF",
    0.9999: "#58A6FF",
    0.99999: "#3FB950",
}


def _apply_theme() -> None:
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": BORDER,
        "axes.labelcolor": TEXT,
        "text.color": TEXT,
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "grid.color": GRID,
        "grid.alpha": 0.4,
        "grid.linewidth": 0.5,
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Inter", "Helvetica Neue", "Segoe UI", "Arial", "DejaVu Sans",
        ],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "figure.titlesize": 16,
        "figure.titleweight": "bold",
        "legend.facecolor": SURFACE,
        "legend.edgecolor": BORDER,
        "legend.fontsize": 9,
        "legend.framealpha": 0.9,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": BG,
        "savefig.edgecolor": "none",
        "savefig.pad_inches": 0.3,
    })


def scan_yolo_dataset(
    data_yaml_path: Path,
) -> tuple[dict[int, str], dict[str, Counter], int]:
    with open(data_yaml_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    dataset_root = data_yaml_path.parent
    if isinstance(cfg["names"], list):
        names = dict(enumerate(cfg["names"]))
    else:
        names = {int(k): v for k, v in cfg["names"].items()}

    split_counts: dict[str, Counter] = {}
    total_images = 0

    for split in ("train", "val"):
        label_dir = dataset_root / "labels" / split
        if not label_dir.exists():
            logger.warning("Label directory not found: %s", label_dir)
            split_counts[split] = Counter()
            continue

        counts: Counter = Counter()
        n_images = 0
        for label_path in label_dir.iterdir():
            if not label_path.is_file() or label_path.suffix.lower() != ".txt":
                continue
            n_images += 1
            content = label_path.read_text(encoding="utf-8").strip()
            if not content:
                continue
            for line in content.splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    counts[int(parts[0])] += 1

        split_counts[split] = counts
        total_images += n_images

    return names, split_counts, total_images


def _total_counts(split_counts: dict[str, Counter]) -> Counter:
    total: Counter = Counter()
    for counts in split_counts.values():
        total += counts
    return total


def effective_number(n: float, beta: float) -> float:
    if beta < 1e-12:
        return 1.0
    if abs(beta - 1.0) < 1e-12:
        return float(n)
    return (1.0 - beta ** n) / (1.0 - beta)


def compute_class_weights(counts: dict[int, int], beta: float = 0.9999) -> dict[int, float]:
    eff = {c: effective_number(n, beta) for c, n in counts.items()}
    inv = {c: 1.0 / max(e, 1e-12) for c, e in eff.items()}
    total = sum(inv.values())
    nc = len(counts)
    return {c: (w / total) * nc for c, w in inv.items()}


def compute_rfs_factors(
    counts: dict[int, int],
    total_images: int,
    threshold: float | None = None,
) -> dict[int, float]:
    freqs = {c: n / max(total_images, 1) for c, n in counts.items()}
    if threshold is None:
        sorted_f = sorted(freqs.values())
        threshold = sorted_f[len(sorted_f) // 2] if sorted_f else 0.01
    factors = {}
    for c, f in freqs.items():
        factors[c] = max(1.0, math.sqrt(threshold / max(f, 1e-12)))
    return factors


def plot_class_distribution(
    names: dict[int, str],
    counts: Counter,
    output_dir: Path,
) -> None:
    sorted_classes = sorted(names.keys(), key=lambda c: counts.get(c, 0))
    labels = [names[c] for c in sorted_classes]
    values = np.array([counts.get(c, 0) for c in sorted_classes], dtype=float)
    total = values.sum()

    log_vals = np.log10(np.clip(values, 1, None))
    norm = (log_vals.max() - log_vals) / max(log_vals.max() - log_vals.min(), 1e-6)
    colors = [SEVERITY_CMAP(v) for v in norm]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(
        range(len(labels)), values, color=colors,
        edgecolor=BORDER, linewidth=0.6, height=0.72,
    )

    ax.set_xscale("log")
    ax.set_xlim(left=50, right=values.max() * 2.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)

    for i, (_bar, val) in enumerate(zip(bars, values)):
        pct = (val / total) * 100 if total > 0 else 0
        ax.text(
            val * 1.15, i, f" {val:,.0f}  ({pct:.1f}%)",
            va="center", fontsize=9, color=TEXT_MUTED, fontweight="medium",
        )

    ax.set_xlabel("Instance Count (log scale)")
    ax.set_title(
        "Class Distribution",
        pad=16, fontsize=15, fontweight="bold",
    )
    ax.text(
        0.5, 1.02,
        f"{total:,.0f} total instances  -  Imbalance ratio: {values.max():.0f} : {max(values.min(), 1):.0f}",
        transform=ax.transAxes, ha="center", fontsize=10, color=TEXT_MUTED,
    )

    fig.tight_layout()
    _save(fig, output_dir / "01_class_distribution")


def plot_imbalance_matrix(
    names: dict[int, str],
    counts: Counter,
    output_dir: Path,
) -> None:
    sorted_ids = sorted(names.keys())
    n = len(sorted_ids)
    labels = [names[c] for c in sorted_ids]
    vals = np.array([max(counts.get(c, 0), 1) for c in sorted_ids], dtype=float)

    ratio_matrix = vals[:, None] / vals[None, :]

    fig, ax = plt.subplots(figsize=(10, 8.5))
    norm = LogNorm(vmin=ratio_matrix.min(), vmax=ratio_matrix.max())

    diverge_colors = ["#FF7B72", "#FFA657", "#E6EDF3", "#79C0FF", "#58A6FF"]
    div_cmap = LinearSegmentedColormap.from_list("div", diverge_colors)

    im = ax.imshow(
        ratio_matrix, cmap=div_cmap, norm=norm,
        aspect="auto", interpolation="nearest",
    )

    for i in range(n):
        for j in range(n):
            r = ratio_matrix[i, j]
            txt = f"{r:.0f}x" if r >= 10 else f"{r:.1f}x"
            if i == j:
                txt = "1:1"
            text_col = "#0D1117" if 0.3 < norm(r) < 0.7 else TEXT
            ax.text(
                j, i, txt, ha="center", va="center",
                fontsize=7 if n > 8 else 8, color=text_col, fontweight="medium",
            )

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.tick_params(length=0)

    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label("Count Ratio (row / column)", fontsize=10, color=TEXT_MUTED)
    cbar.ax.tick_params(colors=TEXT_MUTED, labelsize=9)

    ax.set_title(
        "Pairwise Class Imbalance Ratios",
        pad=16, fontsize=15, fontweight="bold",
    )

    fig.tight_layout()
    _save(fig, output_dir / "02_imbalance_matrix")


def plot_train_val_split(
    names: dict[int, str],
    split_counts: dict[str, Counter],
    output_dir: Path,
) -> None:
    sorted_ids = sorted(names.keys())
    labels = [names[c] for c in sorted_ids]
    n = len(sorted_ids)
    x = np.arange(n)
    width = 0.35

    train_vals = np.array([split_counts.get("train", Counter()).get(c, 0) for c in sorted_ids], dtype=float)
    val_vals = np.array([split_counts.get("val", Counter()).get(c, 0) for c in sorted_ids], dtype=float)

    train_total = train_vals.sum()
    val_total = val_vals.sum()
    train_pct = (train_vals / max(train_total, 1)) * 100
    val_pct = (val_vals / max(val_total, 1)) * 100

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.bar(
        x - width / 2, train_pct, width, label=f"Train ({train_total:,.0f})",
        color=TRAIN_COLOR, edgecolor=BORDER, linewidth=0.5, alpha=0.88,
    )
    ax.bar(
        x + width / 2, val_pct, width, label=f"Val ({val_total:,.0f})",
        color=VAL_COLOR, edgecolor=BORDER, linewidth=0.5, alpha=0.88,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Proportion within Split (%)")
    ax.set_ylim(0, max(train_pct.max(), val_pct.max()) * 1.25)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    ax.legend(
        loc="upper right", framealpha=0.9, fontsize=10,
        facecolor=SURFACE, edgecolor=BORDER,
    )
    ax.set_title(
        "Train / Validation Split Distribution",
        pad=16, fontsize=15, fontweight="bold",
    )

    fig.tight_layout()
    _save(fig, output_dir / "03_train_val_split")


def plot_effective_number(
    names: dict[int, str],
    counts: Counter,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))

    n_range = np.logspace(0, 6, 500)
    betas = [0.9, 0.99, 0.999, 0.9999, 0.99999]

    for beta in betas:
        e_vals = np.array([effective_number(n, beta) for n in n_range])
        ax.plot(
            n_range, e_vals, color=BETA_COLORS[beta],
            linewidth=2 if beta == 0.9999 else 1.2,
            alpha=1.0 if beta == 0.9999 else 0.55,
            label=f"beta = {beta}",
            zorder=3 if beta == 0.9999 else 2,
        )

    sorted_ids = sorted(counts.keys(), key=lambda c: counts[c])
    for c in sorted_ids:
        n = counts[c]
        e = effective_number(n, 0.9999)
        ax.plot(
            n, e, "o", color="#FF7B72", markersize=6,
            markeredgecolor=TEXT, markeredgewidth=0.8, zorder=5,
        )
        ax.annotate(
            names[c], (n, e),
            textcoords="offset points", xytext=(8, 4),
            fontsize=7.5, color=TEXT_MUTED,
            arrowprops={"arrowstyle": "-", "color": GRID, "lw": 0.5},
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Samples (n)")
    ax.set_ylabel("Effective Number of Samples  E_n")
    ax.grid(True, which="both", alpha=0.15)
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper left", framealpha=0.9, fontsize=9,
        facecolor=SURFACE, edgecolor=BORDER, title="Cui et al. (CVPR 2019)",
        title_fontsize=9,
    )
    ax.set_title(
        "Effective Number of Samples - Diminishing Returns of Data",
        pad=16, fontsize=15, fontweight="bold",
    )
    ax.text(
        0.5, 1.02,
        "Each additional sample contributes less as class size grows  -  beta = 0.9999 (recommended)",
        transform=ax.transAxes, ha="center", fontsize=10, color=TEXT_MUTED,
    )

    fig.tight_layout()
    _save(fig, output_dir / "04_effective_number")


def plot_rebalancing_preview(
    names: dict[int, str],
    counts: Counter,
    total_images: int,
    output_dir: Path,
) -> None:
    rfs = compute_rfs_factors(counts, total_images)
    weights = compute_class_weights(counts, beta=0.9999)

    sorted_ids = sorted(names.keys(), key=lambda c: counts.get(c, 0))
    labels = [names[c] for c in sorted_ids]
    original = np.array([counts.get(c, 0) for c in sorted_ids], dtype=float)
    balanced = np.array([counts.get(c, 0) * rfs.get(c, 1.0) for c in sorted_ids], dtype=float)
    factors = [rfs.get(c, 1.0) for c in sorted_ids]
    w_vals = [weights.get(c, 1.0) for c in sorted_ids]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [3, 2]})

    y = np.arange(len(labels))
    height = 0.35

    ax1.barh(
        y + height / 2, original, height, label="Original",
        color="#8B949E", edgecolor=BORDER, linewidth=0.5, alpha=0.7,
    )
    ax1.barh(
        y - height / 2, balanced, height, label="After RFS",
        color=ACCENT, edgecolor=BORDER, linewidth=0.5, alpha=0.88,
    )

    for i, (_orig, bal, f) in enumerate(zip(original, balanced, factors)):
        if f > 1.01:
            ax1.text(
                bal * 1.08, i - height / 2, f"x{f:.1f}",
                va="center", fontsize=8, color="#3FB950", fontweight="bold",
            )

    ax1.set_xscale("log")
    ax1.set_xlim(left=50)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=10)
    ax1.set_xlabel("Instance Count (log scale)")
    ax1.grid(axis="x", alpha=0.15)
    ax1.set_axisbelow(True)
    ax1.legend(
        loc="lower right", framealpha=0.9, fontsize=10,
        facecolor=SURFACE, edgecolor=BORDER,
    )
    ax1.set_title("Repeat Factor Sampling Effect", fontsize=13, fontweight="bold", pad=12)

    w_arr = np.array(w_vals)
    log_w = np.log10(np.clip(w_arr, 0.01, None))
    norm_w = (log_w - log_w.min()) / max(log_w.max() - log_w.min(), 1e-6)
    w_colors = [SEVERITY_CMAP(v) for v in norm_w]

    ax2.barh(y, w_arr, color=w_colors, edgecolor=BORDER, linewidth=0.5, height=0.6)
    for i, w in enumerate(w_arr):
        ax2.text(
            w + w_arr.max() * 0.03, i, f"{w:.2f}",
            va="center", fontsize=9, color=TEXT_MUTED,
        )
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=10)
    ax2.set_xlabel("Class-Balanced Loss Weight")
    ax2.grid(axis="x", alpha=0.15)
    ax2.set_axisbelow(True)
    ax2.set_title("Cui et al. Loss Weights (beta=0.9999)", fontsize=13, fontweight="bold", pad=12)

    fig.suptitle(
        "Rebalancing Strategy Preview - RFS + Class-Balanced Focal Loss",
        fontsize=16, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    _save(fig, output_dir / "05_rebalancing_preview")


def _save(fig: plt.Figure, stem: Path) -> None:
    for ext in ("png", "pdf"):
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path)
        logger.info("Saved: %s", path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    setup_config()

    try:
        import hydra
        from omegaconf import DictConfig
    except ImportError as exc:
        raise ImportError("hydra-core is required for visualization. Run: pip install hydra-core") from exc

    @hydra.main(version_base=None, config_path="../../../configs", config_name="visualize")
    def _run(cfg: DictConfig) -> None:
        args: VisualizationConfig = VisualizationConfig(
            data_yaml=cfg.visualization.data_yaml,
            output_dir=cfg.visualization.output_dir,
        )

        data_yaml = Path(args.data_yaml).resolve()
        if not data_yaml.exists():
            raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

        output_dir = Path(args.output_dir).resolve() if args.output_dir else (PROJECT_ROOT / "outputs" / "figures").resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        _apply_theme()

        logger.info("Scanning dataset: %s", data_yaml)
        names, split_counts, total_images = scan_yolo_dataset(data_yaml)
        total = _total_counts(split_counts)

        logger.info("Classes: %d  |  Total images: %d  |  Total instances: %d",
                    len(names), total_images, sum(total.values()))
        for cid in sorted(names.keys()):
            logger.info("  %2d | %-20s | %7d", cid, names[cid], total.get(cid, 0))

        logger.info("Generating figures...")
        plot_class_distribution(names, total, output_dir)
        plot_imbalance_matrix(names, total, output_dir)
        plot_train_val_split(names, split_counts, output_dir)
        plot_effective_number(names, total, output_dir)
        plot_rebalancing_preview(names, total, total_images, output_dir)

        logger.info("All %d figures saved to: %s", 5, output_dir)

    _run()


if __name__ == "__main__":
    main()
