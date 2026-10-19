"""Plot online, all-minibatch diagnostics; no model or dataset required."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="training output directory")
    parser.add_argument("--group", default="all_shared")
    args = parser.parse_args()
    folder = Path(args.run) / "gradient_conflict"
    with (folder / "epochs.csv").open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["group"] == args.group]
    if not rows:
        raise SystemExit(f"No rows for {args.group}")
    x = np.array([int(r["epoch"]) for r in rows])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")
    specs = [("conflict_rate", "Conflicting minibatches (%)", 100),
             ("mean_cosine", "Mean gradient cosine", 1),
             ("mean_magnitude_similarity", "Mean magnitude similarity", 1),
             ("epoch_cancellation_fraction", "Cancellation fraction (%)", 100)]
    for ax, (key, label, scale) in zip(axes.flat, specs):
        values = [float(r[key])*scale if r[key] else np.nan for r in rows]
        # Different scope during encoder freeze: show as a separate segment.
        for frozen, color, name in ((1, "#92999F", "Encoder frozen"),
                                    (0, "#245B78", "Encoder trainable")):
            y = np.array(values)
            y[[int(r["encoder_frozen"]) != frozen for r in rows]] = np.nan
            if np.isfinite(y).any():
                ax.plot(x, y, "o-", color=color, ms=3, lw=1.5, label=name)
        ax.set(xlabel="Epoch", ylabel=label)
        ax.grid(axis="y", alpha=.18)
        if key == "mean_cosine":
            ax.axhline(0, color="#A84E4E", lw=.8, ls="--")
            ax.set_ylim(-1, 1)
        else:
            ax.set_ylim(0, 1 if scale == 1 else 100)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(handles, labels, frameon=False, fontsize=8)
    fig.suptitle(f"Cell–Tissue training gradients / {args.group}")
    for extension in ("png", "pdf", "svg"):
        fig.savefig(folder / f"epoch_conflict_{args.group}.{extension}", dpi=300)
    plt.close(fig)
    print(f"[saved] {folder}/epoch_conflict_{args.group}.[png/pdf/svg]")


if __name__ == "__main__":
    main()
