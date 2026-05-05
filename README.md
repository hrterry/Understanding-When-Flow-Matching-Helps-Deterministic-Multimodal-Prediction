# Anonymous Reproducibility Repository

This repository is prepared for anonymous peer review and reproducibility only.  
It intentionally excludes paper narrative, author identity, and institutional information.

## 1. Environment

- OS: Linux (recommended) or Windows (path adjustments may be needed)
- Python: `3.10` (recommended)
- GPU: NVIDIA GPU with CUDA support (training scripts default to GPU execution)

Choose one setup option:

### A. Conda (recommended)

```bash
conda env create -f environment.yml
conda activate PathFlow
```

### B. Pip

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Data and Directory Layout

Training scripts expect the following structure:

```text
<SOURCE_DATAROOT>/
  <DATASET>/
    adata/
      <sample_id>.h5ad
    splits/
      train_0.csv
      test_0.csv
      ...
    var_200genes.json   # or another gene-list JSON containing a "genes" field

<EMBED_DATAROOT>/
  <DATASET>/
    <FEATURE_ENCODER>/
      fp32/
        <sample_id>.h5
```

Requirements:

- `splits/train_i.csv` and `splits/test_i.csv` must contain a `sample_id` column
- `var_*.json` must follow `{"genes": [...]}`
- `<sample_id>.h5` and `<sample_id>.h5ad` must match by sample ID

### Download HEST-1k and HEST-Bench from Hugging Face

Authenticate first (some assets may be gated):

```bash
huggingface-cli login
```

Then download datasets:

```bash
# HEST-1k
huggingface-cli download MahmoodLab/hest \
  --repo-type dataset \
  --local-dir <SOURCE_DATAROOT>

# HEST-Bench
huggingface-cli download MahmoodLab/hest-bench \
  --repo-type dataset \
  --local-dir <SOURCE_DATAROOT>
```

Python alternative (same repository IDs):

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="MahmoodLab/hest",
    repo_type="dataset",
    local_dir="<SOURCE_DATAROOT>",
)
snapshot_download(
    repo_id="MahmoodLab/hest-bench",
    repo_type="dataset",
    local_dir="<SOURCE_DATAROOT>",
)
```

### Download the pathology foundation model (`uni_v1_official`)

For `--feature_encoder uni_v1_official`, download UNI weights from:
`[MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI)`

```bash
# Download UNI checkpoint to a local weights directory
huggingface-cli download MahmoodLab/UNI \
  pytorch_model.bin \
  --local-dir <WEIGHTS_ROOT>/uni
```

Python alternative:

```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="MahmoodLab/UNI",
    filename="pytorch_model.bin",
    local_dir="<WEIGHTS_ROOT>/uni",
)
```

### Generate `var_200genes.json` with script

If your dataset folder does not already contain `var_200genes.json`, generate it via:

```bash
python app/flow/generate_gene_panel.py \
  --source_dataroot <SOURCE_DATAROOT> \
  --dataset SKCM \
  --top-k 200 \
  --output-name var_200genes.json
```

This will scan `<SOURCE_DATAROOT>/<DATASET>/adata/*.h5ad` and write:
`<SOURCE_DATAROOT>/<DATASET>/var_200genes.json`.

## 3. Main Method Reproduction

Run from repository root:

```bash
python app/flow/train.py \
  --source_dataroot <SOURCE_DATAROOT> \
  --embed_dataroot <EMBED_DATAROOT> \
  --save_dir results_dir \
  --datasets SKCM \
  --feature_encoder uni_v1_official \
  --gene_list var_200genes.json \
  --seed 1 \
  --deterministic
```

Notes:

- `--deterministic` improves reproducibility (may reduce throughput)
- Use `--split-only 0` to run a single split
- For a quick smoke test, reduce `--epochs` and/or dataset scope

## 4. Baseline Reproduction (PCA + Ridge)

```bash
python app/flow/pca_ridge_baseline.py \
  --source_dataroot <SOURCE_DATAROOT> \
  --embed_dataroot <EMBED_DATAROOT> \
  --save_dir results_dir/baseline_pca_ridge \
  --datasets SKCM \
  --feature_encoder uni_v1_official \
  --gene_list var_200genes.json \
  --seed 1 \
  --deterministic
```

## 5. Output Artifacts

Main method outputs (timestamped under `--save_dir`):

- `<save_dir>/<timestamp>/<dataset>/config.json`
- `<save_dir>/<timestamp>/<dataset>/results_kfold.json`
- `<save_dir>/<timestamp>/<dataset>/test_results_all_splits.csv`
- `<save_dir>/<timestamp>/<dataset>/split*/checkpoints/best.pth`
- `<save_dir>/<timestamp>/<dataset>/split*/test_metrics.csv`

Baseline outputs:

- `<save_dir>/<dataset>/results_kfold.json`
- `<save_dir>/<dataset>/test_results_all_splits.csv`
- `<save_dir>/<dataset>/split*/predictions.npz`

## 6. Reproducibility Recommendations

- Fix random seed: `--seed <int>`
- Enable deterministic mode: `--deterministic`
- Log full commands, dependency versions, and runtime environment
- Avoid mixing CUDA/PyTorch versions across runs
- Do not modify official split files in `splits/*.csv` during reproduction

## 7. License

All rights reserved during anonymous review.  
A final license statement will be released after publication decisions.

This repository is built upon upstream open-source pathology and spatial transcriptomics toolchains,
including `[mahmoodlab/HEST](https://github.com/mahmoodlab/HEST)`.