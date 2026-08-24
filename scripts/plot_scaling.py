"""Plot scaling-law ablation charts from captured pretrain logs.

Reads the ``tee``-captured stdout of ``scripts/pretrain.py`` for each tier and
produces one figure with two panels:

  1. validation loss vs tokens seen, one curve per model size (log-x)
  2. final validation loss vs non-embedding parameters (log-log), with a fitted
     power law L = a * N^(-b)

Validation loss comes from the ``val ppl`` lines (loss = ln(ppl)), so the runs
must have been launched with ``--eval-every``. A log with no val-ppl lines falls
back to the noisier train-loss lines, and the script reports when it does.

    python scripts/plot_scaling.py --log-dir out/scaling --tiers nano mini small
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_config  # noqa: E402

TRAIN_RE = re.compile(r"step\s+(\d+)\s+loss\s+([\d.]+)")
VAL_RE = re.compile(r"step\s+(\d+)\s+val ppl\s+([\d.]+)")


def parse_log(path: Path) -> tuple[list[int], list[float], str]:
    """Return (steps, losses, source) where source is 'val' or 'train'."""
    text = path.read_text(encoding="utf-8")
    val = [(int(s), math.log(float(p))) for s, p in VAL_RE.findall(text)]
    if val:
        steps, losses = zip(*val)
        return list(steps), list(losses), "val"
    train = [(int(s), float(x)) for s, x in TRAIN_RE.findall(text)]
    if not train:
        raise ValueError(f"{path}: no 'loss' or 'val ppl' lines found")
    steps, losses = zip(*train)
    return list(steps), list(losses), "train"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Plot scaling-law charts from pretrain logs.")
    ap.add_argument("--log-dir", default="out/scaling", help="dir with <tier>.log files")
    ap.add_argument("--tiers", nargs="+", default=["nano", "mini", "small"])
    ap.add_argument("--out", default=None, help="output png (default <log-dir>/scaling_laws.png)")
    args = ap.parse_args(argv)

    log_dir = Path(args.log_dir)
    out_path = Path(args.out) if args.out else log_dir / "scaling_laws.png"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5))
    final_points: list[tuple[float, float]] = []  # (non-embedding params, final loss)

    for tier in args.tiers:
        log_path = log_dir / f"{tier}.log"
        if not log_path.exists():
            print(f"skipping {tier}: {log_path} not found")
            continue
        cfg = get_config(tier)
        steps, losses, source = parse_log(log_path)
        if source == "train":
            print(f"note: {tier} has no val-ppl lines (run with --eval-every); using train loss")

        tokens = [s * cfg.global_batch_tokens for s in steps]
        n = cfg.param_count().non_embedding
        ax1.plot(tokens, losses, marker="o", markersize=3, linewidth=1.2,
                 label=f"{tier} ({n / 1e6:.1f}M non-emb)")
        final_points.append((n, losses[-1]))
        print(f"{tier}: {len(steps)} points, final {source} loss {losses[-1]:.4f} "
              f"at {tokens[-1] / 1e9:.2f}B tokens")

    if not final_points:
        print("no logs found; nothing to plot")
        return 1

    ax1.set_xscale("log")
    ax1.set_xlabel("tokens seen")
    ax1.set_ylabel("val loss")
    ax1.set_title("Loss vs tokens, per model size")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ns = np.array([p[0] for p in final_points], dtype=float)
    ls = np.array([p[1] for p in final_points], dtype=float)
    ax2.plot(ns, ls, "o", markersize=7, color="tab:red")

    # Power-law fit L = a * N^(-b): a straight line in log-log space.
    if len(final_points) >= 2:
        b, log_a = np.polyfit(np.log(ns), np.log(ls), 1)
        grid = np.geomspace(ns.min() * 0.8, ns.max() * 1.25, 100)
        ax2.plot(grid, np.exp(log_a) * grid**b, "--", color="gray",
                 label=f"fit: L = {math.exp(log_a):.2f} * N^({b:.3f})")
        ax2.legend()

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("non-embedding parameters")
    ax2.set_ylabel("final val loss")
    ax2.set_title("Scaling law: loss vs model size")
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
