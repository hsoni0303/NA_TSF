#!/usr/bin/env bash
# Full evaluation under the fixed protocol. Run on a server, not a laptop.
#   DATASETS="electricity" SEEDS=5 bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
DATASETS=${DATASETS:-"electricity traffic"}
SEEDS=${SEEDS:-5}
SETTINGS="none:0.0 const:0.1 const:0.3 missing:0.1 missing:0.3 gaussian:0.1 gaussian:0.3"
mkdir -p results
bash scripts/get_baselines.sh

for ds in $DATASETS; do
  # 1) NA-TSF: grid search on seed 0, then the best config per setting over $SEEDS seeds
  python na_tsf/train.py --dataset "$ds" --tune 1 --verbose 0 --out "results/tune_$ds.csv"
  python na_tsf/analyze.py "results/tune_$ds.csv" --commands "$SEEDS" --out "results/eval_$ds.csv" | bash
  python na_tsf/analyze.py "results/eval_$ds.csv" | tee "results/summary_$ds.txt"

  # 2) Baselines (official RobustTSF code), repeated $SEEDS times per setting
  short=$([ "$ds" = electricity ] && echo ele || echo traffic)
  out="results/baselines_$ds.csv"
  [ -f "$out" ] || echo "method,noise,ratio,run,mae_best,mse_best,mae_last" > "$out"
  for s in $SETTINGS; do
    noise=${s%%:*}; ratio=${s##*:}
    for method in train_noisy train_noisy_weighting; do
      for run in $(seq 0 $((SEEDS - 1))); do
        log=$(cd baselines/robusttsf && python "$method.py" --loss mae --dataset "$short" \
              --ano_type "$noise" --ano_ratio "$ratio" 2>&1)
        get() { echo "$log" | grep "$1" | tail -1 | awk '{print $NF}'; }
        echo "$method,$noise,$ratio,$run,$(get 'val mae (best)'),$(get 'val mse (best)'),$(get 'val mae (last)')" >> "$out"
      done
    done
  done
done
