#!/usr/bin/env bash
# Section 4.3: routing on real features and the time dose-response on synthetic data.
set -e
mkdir -p cache runs
for spec in LUNG:0 LUNG:1 COAD:0 COAD:1 PAAD:0 PAAD:1 LYMPH_IDC:0 LYMPH_IDC:1 PRAD:0; do
  ds=${spec%:*}; sp=${spec#*:}; f=cache/${ds}_${sp}.npz
  python analysis/cache_features.py --config configs/latent.yaml --dataset "$ds" --split "$sp" --out "$f"
  for seed in 0 1 2; do
    for arm in reg_tok fm fm_wperm fm_paired; do
      python analysis/simulation.py --arm $arm --data "$f" --seed $seed --out runs/routing.jsonl
    done
  done
done
for seed in 0 1 2; do
  for tm in 0.3 0.5 0.7 0.85 0.95 0.99; do
    for loss in v x; do
      python analysis/simulation.py --arm fm_paired --t-max $tm --loss $loss --seed $seed --out runs/time.jsonl
    done
  done
done
