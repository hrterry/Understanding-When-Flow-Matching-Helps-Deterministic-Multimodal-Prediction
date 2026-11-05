#!/usr/bin/env bash
# Table 1b: controlled ablations on IDC-80k. Figure 5: --set prediction=v.
set -e
C="--config configs/genespace.yaml --dataset IDC_all --seed ${1:-1}"
python train.py $C
python train.py $C --set xt_mode=clean
python train.py $C --set xt_mode=permuted
python train.py $C --set time=none
python train.py $C --set backbone=vanilla
python train.py $C --set backbone=vanilla xt_mode=none
python train.py $C --set prediction=v
