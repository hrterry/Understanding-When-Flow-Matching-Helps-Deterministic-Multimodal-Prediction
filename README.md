# When Does Flow Matching Help Deterministic Multimodal Prediction? A Spatial Transcriptomics Study

**Haoran Xu, Wei Yuan, Xiao Han** · NeurIPS 2026

[Project page](https://hrterry.github.io/Understanding-When-Flow-Matching-Helps-Deterministic-Multimodal-Prediction/)

Generative models are increasingly used for prediction tasks with one correct answer. We ask what flow
matching adds over regression in histology-to-spatial-transcriptomics (ST) prediction. Its benefit does
not come from sampling diversity: the noisy intermediate target states act as target-side supervision,
when they are condition-aligned, time-aware and selectively integrated, mainly where paired data are scarce.

This repository contains only the model and the experiments of the paper.

## Install

```bash
pip install -r requirements.txt
```

Python 3.10+, PyTorch with CUDA.

## Data

We use [HEST-1k / HEST-Bench](https://huggingface.co/datasets/MahmoodLab/hest) with
[UNI](https://huggingface.co/MahmoodLab/UNI) patch embeddings (both gated on Hugging Face).

```text
data/hest/<DATASET>/adata/<sample_id>.h5ad
data/hest/<DATASET>/splits/{train,test}_<k>.csv        # column: sample_id
data/hest/<DATASET>/var_200genes.json                  # {"genes": [...]}
data/embed_dataroot/<DATASET>/uni_v1_official/fp32/<sample_id>.h5   # embeddings, coords, barcodes
```

Embeddings follow the HEST-Bench extraction (224×224 patches at 0.5 µm/px, one per spot). The IDC cohorts
use `data/hest1k_datasets/` in the same layout. Paths are set in `configs/*.yaml`.

## Quick start

```bash
python train.py --config configs/latent.yaml --dataset COAD --seed 1
```

This trains every predefined split and writes `results/COAD/latent_seed1/summary.json`. Any setting can be
overridden, for example `--set formulation=regression` or `--set eval_every=5`.

| Config | Used for |
|---|---|
| `configs/latent.yaml` | Tables 1a and 2: flow in the latent space of a jointly trained VAE |
| `configs/genespace.yaml` | IDC Table 1b and scaling: flow directly on genes |
| `configs/genespace_local.yaml` | Mechanism analyses: gene space, local groups of 128 spots |

## Reproducing the paper

| Result | Command |
|---|---|
| Table 2, matched comparison (Section 4.3) | `bash scripts/table2.sh 1` (seeds 1, 40, 2026) |
| Table 1b ablations, Figure 5 | `bash scripts/ablations.sh` |
| Context size, capacity, Qwen-style backbone (Appendix E) | `bash scripts/idc_extended.sh` |
| Routing and time experiments (Section 4.3) | `bash scripts/simulation.sh` |
| Routing interventions | `python analysis/routing.py --run <split dir> --split 0` |
| What the model reads from x_t | `python analysis/probe_xt.py --run <split dir> --split 0` |
| Repeated sampling | `python analysis/determinism.py --run <split dir> --split 0` |

As in the STFlow pipeline, the reported epoch of each fold is the one with the best PCC on the evaluation
slides. PCC-k is the mean over the k best-predicted genes. Spatial proteomics runs (Appendix E) need the
Visium PEX and CRC-CODEX cohorts prepared in the same layout, with `--set gene_list=<panel>.json`.

## Model

`flowst/model.py` holds the whole model. A sequence of histology tokens, gene embeddings and noisy ST-state
tokens passes through self-attention with one learned gate per quadrant, applied to the logits before
softmax, and the clean endpoint is read from the histology tokens. Noisy tokens carry no spot index, so each
histology token reads them as a shared expression context of the group.

```text
flowst/model.py     gated flow model, Context-Matched Regression, gene VAE
flowst/qwen.py      Qwen-style gated backbone
flowst/flow.py      x-prediction loss, v-prediction comparison, Euler sampling
flowst/data.py      slides, splits, shuffled and local spot groups
flowst/metrics.py   PCC-k, Spearman, MSE, MAE, LISA
train.py            training and evaluation
baseline_pca_ridge.py   Original Regression (HEST-Bench ridge)
analysis/           routing, x_t probes, repeated sampling, simulation
scripts/            one script per group of experiments
docs/               project page
```

## Verification

The code was checked against the runs behind the paper. Loading the original checkpoints into this model
gives the reported test metrics:

| Dataset (split) | PCC here | PCC reported |
|---|---|---|
| COAD (0) | 0.4758 | 0.4758 |
| LUNG (0) | 0.6645 | 0.6645 |
| PAAD (1) | 0.5540 | 0.5540 |
| LYMPH (0) | 0.4291 | 0.4291 |
| IDC-80k | 0.6603 | 0.6604 |
| IDC, local M=128 | 0.6856 | 0.6856 |

Retraining LUNG from scratch with this code (150 epochs, evaluation every 5) gives 0.629, against 0.635,
0.629 and 0.631 for the original seeds. The original experiment code also carried gene-rank and
non-negativity options; they did not affect the results and are not included here.

## Citation

```bibtex
@inproceedings{xu2026flowdeterministic,
  title     = {When Does Flow Matching Help Deterministic Multimodal Prediction? A Spatial Transcriptomics Study},
  author    = {Xu, Haoran and Yuan, Wei and Han, Xiao},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

Data from HEST-1k / HEST-Bench and features from UNI (Mahmood Lab). The evaluation pipeline follows
[STFlow](https://github.com/Graph-and-Geometric-Learning/STFlow).

## License

MIT
