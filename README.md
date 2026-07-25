# Heart Disease Classification - PyTorch Experiment Framework

A structured machine-learning experiment comparing **regularisation methods** and **optimisers** on the Heart Disease UCI dataset, implemented end-to-end in PyTorch inside a Jupyter notebook.

This project is a graded lab assignment for the course *Statistisches Lernen 2* (MSc FH Kufstein, 2nd semester).

---

## What we are doing - and why

### Problem
Binary classification: given 13 clinical features (age, cholesterol, resting blood pressure, etc.), predict whether a patient has heart disease (`target = 1`) or not (`target = 0`).

### Approach: Two-phase, multi-seed CV ablation study
Rather than running a brute-force grid search over all regulariser × optimiser combinations (which conflates effects), the experiment uses a **controlled two-phase design**:

| Phase | Fixed | Varied | Goal |
|-------|-------|--------|------|
| **Phase 1** | Optimizer = Adam | All 6 regularisers | Select regulariser by mean CV ROC-AUC |
| **Phase 2** | Regulariser = Phase 1 winner | All 5 optimisers | Select optimiser by mean CV ROC-AUC |
| **Final** | Best CV configuration | - | SWA retraining on development data + one test evaluation |

Each configuration uses 5 stratified folds and three seeds (15 fits). Standard deviations expose sampling and initialization sensitivity. A stratified 15% test set remains untouched until final evaluation.

### Why SWA?
As a beyond-syllabus technique, **Stochastic Weight Averaging** (Izmailov et al., 2018) is applied as the final step. SWA averages model weights across the last 25 % of training epochs and is evaluated against the base model rather than assumed to improve generalisation.

---

## Repository structure

```
.
├── heart_disease_experiment.ipynb   # Main notebook - all experiments and visualisations
├── requirements.txt                 # Python dependencies (pip)
├── .vscode/settings.json            # VS Code / Jupyter kernel config
├── KaggleToken.py                   # ← NOT committed (git-ignored). See setup below.
└── data/
    └── heart.csv                    # ← Bundled dataset. Kaggle credentials are optional
                                     #   (notebook falls back to this file automatically).
```

---

## Setup

### Prerequisites
- Python 3.12
- Git
- Optional: a [Kaggle account](https://www.kaggle.com) for re-downloading the bundled dataset

### 1 - Clone the repository

```powershell
git clone https://github.com/sg-Ikigai/model-factory-heart-diseases.git
cd model-factory-heart-diseases
```

### 2 - Create and activate the virtual environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3 - Install dependencies

Install shared dependencies first. PyTorch is deliberately excluded from `requirements.txt` so this
step cannot replace a CUDA build with the default PyPI build.

```powershell
pip install -r requirements.txt
```

For an NVIDIA GPU compatible with CUDA 12.8:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

For CPU-only:

```powershell
pip install torch
```

### 4 - Kaggle credentials (optional - dataset is bundled)

`data/heart.csv` is included in the repository, so the notebook works out-of-the-box without any Kaggle account.
The Kaggle API is only used when `heart.csv` is absent and credentials are available - useful if you delete the file and want to re-download it.

If you do want to set up Kaggle credentials:

**Option A - `KaggleToken.py` (recommended for local dev)**

Create a file `KaggleToken.py` in the project root (it is git-ignored):

```python
# KaggleToken.py
KAGGLE_API_TOKEN = "KGAT_xxxxxxxxxxxxxxxx"
```

Get your token at [kaggle.com/settings/api](https://www.kaggle.com/settings/api) → *Generate New Token*.

**Option B - Environment variable**

```powershell
$env:KAGGLE_API_TOKEN = "KGAT_xxxxxxxxxxxxxxxx"
```

**Option C - `kaggle.json`**

Download `kaggle.json` from the API settings page and place it at `~/.kaggle/kaggle.json`.

**Option D - Manual download (no Kaggle account required)**

1. Download `heart.csv` from [`ineubytes/heart-disease-dataset`](https://www.kaggle.com/datasets/ineubytes/heart-disease-dataset) and place it at `data/heart.csv`.
2. In the **Configuration** cell of the notebook, set:
   ```python
   USE_LOCAL_CSV = True
   ```
   This skips all Kaggle credential logic entirely.

> **Automatic fallback:** `USE_LOCAL_CSV = False` (the default) tries the Kaggle API first.
> If credentials are missing, invalid, or expired, the notebook automatically falls back to
> `data/heart.csv` on disk - with a warning printed to the cell output. No crash, no data loss.
> Since `heart.csv` is bundled with the repo, the experiment runs end-to-end with no external
> dependencies required.

### 5 - Open the notebook

The workspace points Python and new terminals at `.venv`. After cloning or changing interpreter
settings, run **Developer: Reload Window** once. Existing terminals are not retroactively activated;
open a new terminal after the reload. In the notebook kernel picker, select
**Python Environments → `.venv\Scripts\python.exe`** once. Kernel selection is then persisted.

To verify the selected environment:

```powershell
python -c "import sys, torch; print(sys.executable); print(torch.__version__, torch.cuda.is_available())"
```

Open `heart_disease_experiment.ipynb` and select the kernel **Python 3.12 (.venv)**.  
Run cells top-to-bottom with **Run All** or step through them manually.

The complete model-selection stage performs 165 fits. GPU execution is recommended, although this
small tabular workload also runs on CPU.

---

## Dataset

| Property | Value |
|---|---|
| Source | [ineubytes/heart-disease-dataset](https://www.kaggle.com/datasets/ineubytes/heart-disease-dataset) on Kaggle |
| Raw rows | 1 025 |
| Exact duplicate rows removed | 723 (70.5%) |
| Unique patients used | 302 |
| Resampling | 85% development; 15% untouched test; 5-fold CV × 3 seeds on development |
| Features | 13 clinical features (age, sex, cp, trestbps, chol, …) |
| Target | Binary - `0` = No Disease, `1` = Has Disease |
| Missing values | None |
| Class balance | 49 % / 51 % (nearly balanced) |

---

## Regularisers tested

| Name | Mechanism |
|---|---|
| Baseline | No regularisation |
| L1 | Penalty on absolute weight magnitudes added to loss |
| L2 / Weight Decay | Penalty on squared weight magnitudes via optimiser |
| Dropout | Stochastic neuron deactivation during training |
| Label Smoothing | Softens hard 0/1 targets to reduce overconfident logits |
| BatchNorm + Dropout | BatchNorm before each layer + Dropout |

## Optimisers tested

SGD · SGD with Momentum · Adam · AdamW · RMSprop

All runs share a **CosineAnnealingLR** scheduler so no optimiser is handicapped by a fixed learning rate.

---

## Evaluation metrics

| Metric | Why |
|---|---|
| ROC-AUC | Threshold-independent; primary ranking metric |
| F1-Score | Robust to class imbalance |
| Accuracy | Absolute classification rate |
| Confusion matrix | Exposes false-negative rate - critical for medical classification |
| Precision-Recall curve | More informative than ROC in medical contexts |
| CV standard deviation | Quantifies instability across folds and random seeds |
| Calibration curve | Assesses whether predicted probabilities match observed risk |

---

## Dependencies

See [`requirements.txt`](requirements.txt). Key packages:

| Package | Version |
|---|---|
| torch | ≥ 2.3 (CUDA build recommended) |
| scikit-learn | ≥ 1.5 |
| plotly | ≥ 5.22 |
| kaggle | ≥ 1.6 |
| ipykernel | ≥ 6.29 |
