#!/usr/bin/env bash
# Table 2 and the matched comparison of Section 4.3: flow, CM-Regression and PCA-Ridge on six datasets.
set -e
SEED=${1:-1}
for ds in COAD LYMPH_IDC LUNG PAAD PRAD CCRCC; do
  python train.py --config configs/latent.yaml --dataset "$ds" --seed "$SEED"
  python train.py --config configs/latent.yaml --dataset "$ds" --seed "$SEED" --set formulation=regression
  python baseline_pca_ridge.py --config configs/latent.yaml --dataset "$ds" --seed "$SEED"
done
