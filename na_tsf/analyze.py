"""Summarize result CSVs written by train.py (--out or --tune).

  python na_tsf/analyze.py tune_results_electricity.csv            # best config per setting
  python na_tsf/analyze.py tune_results_electricity.csv --commands 5
                                                                    # 5-seed eval commands for those configs
  python na_tsf/analyze.py eval_electricity.csv                    # mean ± std over seeds
"""

import argparse

import pandas as pd

SETTING = ["dataset", "noise", "ratio"]
CONFIG = ["stl", "context", "hidden", "embed"]
METRICS = ["mae_best", "mse_best", "mae_last", "mse_last"]


def summarize(df):
    """Mean ± std over seeds for every (setting, config), keeping the best config per setting."""
    g = df.groupby(SETTING + CONFIG)[METRICS].agg(["mean", "std", "count"]).reset_index()
    g.columns = [f"{a}_{b}" if b else a for a, b in g.columns]
    best = g.loc[g.groupby(SETTING)["mae_best_mean"].idxmin()]
    return best.sort_values(SETTING).reset_index(drop=True)


def fmt(row, m):
    std = row[f"{m}_std"]
    return f"{row[f'{m}_mean']:.4f}" + ("" if pd.isna(std) else f" ± {std:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv")
    p.add_argument("--commands", type=int, metavar="SEEDS", help="print train.py commands for N seeds")
    p.add_argument("--out", default="eval_results.csv", help="CSV the printed commands write to")
    args = p.parse_args()

    best = summarize(pd.read_csv(args.csv))
    if args.commands:
        for _, r in best.iterrows():
            for seed in range(args.commands):
                print(f"python na_tsf/train.py --verbose 0 --dataset {r.dataset} --ano_type {r.noise} "
                      f"--ano_ratio {r.ratio} --stl_period {r.stl} --stl_context {r.context} "
                      f"--hidden {r.hidden} --embed {r.embed} --seed {seed} --out {args.out}")
    else:
        for _, r in best.iterrows():
            print(f"{r.dataset:<12} {r.noise:<9} {r.ratio:<4} stl={r.stl} ctx={r.context} "
                  f"hid={r.hidden} emb={r.embed} n={int(r.mae_best_count)} | "
                  f"MAE {fmt(r, 'mae_best')}  MSE {fmt(r, 'mse_best')}  (last MAE {fmt(r, 'mae_last')})")
