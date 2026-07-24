# Heart Disease Classification - PyTorch Experiment Framework

A structured machine-learning experiment comparing **regularisation methods** and **optimisers** on the Heart Disease UCI dataset, implemented end-to-end in PyTorch inside a Jupyter notebook.

This project is a graded lab assignment for the course *Statistisches Lernen 2* (MSc FH Kufstein, 2nd semester).

---

## What we are doing - and why

### Problem
Binary classification: given 13 clinical features (age, cholesterol, resting blood pressure, etc.), predict whether a patient has heart disease (`target = 1`) or not (`target = 0`).

### Approach: Two-phase ablation study
Rather than running a brute-force grid search over all regulariser × optimiser combinations (which conflates effects), the experiment uses a **controlled two-phase design**:

| Phase | Fixed | Varied | Goal |
|-------|-------|--------|------|
| **Phase 1** | Optimizer = Adam | All 6 regularisers | Isolate regularisation effect |
| **Phase 2** | Regulariser = best from Phase 1 | All 5 optimisers | Isolate optimiser effect |
| **Final** | Best config from Phase 2 | - | SWA retraining + full evaluation |

This mirrors a controlled experiment (one variable at a time) and is both more interpretable and computationally frugal than an N×M Cartesian product.

### Why SWA?
As a beyond-syllabus technique, **Stochastic Weight Averaging** (Izmailov et al., 2018) is applied as the final step. SWA averages model weights across the last 25 % of training epochs, landing in *flatter* loss-surface minima that generalise better - at zero additional gradient-step cost.

---

## Repository structure

```
.
├── heart_disease_experiment.ipynb   # Main notebook - all experiments and visualisations
├── requirements.txt                 # Python dependencies (pip)
├── .vscode/settings.json            # VS Code / Jupyter kernel config
├── KaggleToken.py                   # ← NOT committed (git-ignored). See setup below.
└── data/                            # ← NOT committed (git-ignored). See setup below.
```

---

## Setup

### Prerequisites
- Python 3.12
- Git
- A [Kaggle account](https://www.kaggle.com) for automatic dataset download

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

> **Important:** The default `pip install torch` installs a CPU-only build.
> Install the CUDA 12.8 build instead if you have an NVIDIA GPU:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

For CPU-only:

```powershell
pip install -r requirements.txt
```

### 4 - Kaggle credentials (dataset auto-download)

The notebook downloads the dataset automatically via the Kaggle API.
You need credentials **once**:

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

**Option D — Manual download (no Kaggle account required)**

1. Download `heart.csv` from [`ineubytes/heart-disease-dataset`](https://www.kaggle.com/datasets/ineubytes/heart-disease-dataset) and place it at `data/heart.csv`.
2. In the **Configuration** cell of the notebook, set:
   ```python
   USE_LOCAL_CSV = True
   ```
   This skips all Kaggle credential logic entirely.

> **Automatic fallback:** If `USE_LOCAL_CSV = False` but credentials are missing or an API token is
> invalid/expired, the notebook automatically falls back to an existing `data/heart.csv` if one is
> present on disk — with a warning printed to the cell output. No crash, no data loss.

### 5 - Open the notebook

In VS Code, open `heart_disease_experiment.ipynb` and select the kernel **Python 3.12 (.venv)**.  
Run cells top-to-bottom with **Run All** or step through them manually.

---

## Dataset

| Property | Value |
|---|---|
| Source | [ineubytes/heart-disease-dataset](https://www.kaggle.com/datasets/ineubytes/heart-disease-dataset) on Kaggle |
| Rows | 1 025 |
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
