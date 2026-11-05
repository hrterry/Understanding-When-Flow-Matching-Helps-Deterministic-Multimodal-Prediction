#!/usr/bin/env bash
# Appendix E: context size, sampling, capacity and the Qwen-style backbone on IDC.
set -e
G="--config configs/genespace.yaml --dataset IDC_all"
L="--config configs/genespace_local.yaml --dataset IDC_all"
for arm in flow regression; do
  python train.py $G --set formulation=$arm M=64
  python train.py $G --set formulation=$arm M=128
  python train.py $L --set formulation=$arm
done
python train.py $L --set hidden_dim=256
python train.py $L --set hidden_dim=256 n_layers=8 n_heads=8
python train.py $L --set hidden_dim=256 n_layers=8 n_heads=8 lr=0.0001
python train.py $L --set hidden_dim=512 n_layers=8 n_heads=8
python train.py $L --set hidden_dim=512 n_layers=8 n_heads=8 lr=0.00005
python train.py $L --set backbone=qwen hidden_dim=256 n_heads=8 lr=0.0001
python train.py $L --set backbone=qwen hidden_dim=384 n_layers=8 n_heads=8 lr=0.00005
