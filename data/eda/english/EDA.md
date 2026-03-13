# NPTEL2020 EDA — Environment & Data Setup Guide
**Project:** Automated Correction of Speech Recordings (Low-Bandwidth Environments)  
**OS:** Windows 11 + WSL2 (Ubuntu)  
**Date:** March 2026  

---

## Required Python Packages
Use the repo-local notebook environment instead of installing packages into the system Python:

```bash
./scripts/setup_notebook_env.sh
```

| Package | Source | Purpose |
|---|---|---|
| `librosa` | pip | Audio loading, mel spectrograms, energy features |
| `soundfile` | apt + pip | Fast WAV header reads without loading PCM |
| `jiwer` | pip | Word Error Rate (WER) and CER computation |
| `pydub` | pip | Audio format conversion (WAV <-> OPUS via ffmpeg) |
| `nltk` | pip | OOV vocabulary analysis (words corpus) |
| `jupyter` | pip | Notebook server |
| `ipykernel` | pip | Python kernel for Jupyter |
| `ipywidgets` | pip | Interactive notebook widgets |
| `numpy` | pip | Numerical computing |
| `pandas` | pip | Tabular data manipulation |
| `matplotlib` | pip | Plotting and visualization |
| `seaborn` | pip | Statistical visualizations |
| `tqdm` | pip | Progress bars for loops |

---

## Step 1 — Check Disk Space
```bash
df -h | grep -v tmpfs
```

**Result:** `D:\` had 1.9 TB free → chosen as data directory.

---

## Step 2 — Create Data Directory
```bash
mkdir -p /mnt/d/nptel2020
cd /mnt/d/nptel2020
```

---

## Step 3 — Install System Packages
```bash
sudo apt-get update -qq
sudo apt-get install -y ffmpeg libsndfile1 python3-pip python3-venv build-essential wget
sudo apt install python3-soundfile
```

---

## Step 4 — Install Python Packages

From the repository root, use the project-managed environment instead of the system Python:
```bash
./scripts/setup_notebook_env.sh
```

---

## Step 5 — Download Dataset (Train Split)

Data is hosted across 20 Zenodo records as split `.tar.gz` parts.  
Run all commands from `/mnt/d/nptel2020`.
```bash
wget https://zenodo.org/record/4590121/files/nptel-train.tar.gz.partaa
wget https://zenodo.org/record/4590910/files/nptel-train.tar.gz.partab
wget https://zenodo.org/record/4590927/files/nptel-train.tar.gz.partac
wget https://zenodo.org/record/4591351/files/nptel-train.tar.gz.partad
wget https://zenodo.org/record/4591367/files/nptel-train.tar.gz.partae
wget https://zenodo.org/record/4591412/files/nptel-train.tar.gz.partaf
wget https://zenodo.org/record/4591424/files/nptel-train.tar.gz.partag
wget https://zenodo.org/record/4591448/files/nptel-train.tar.gz.partah
wget https://zenodo.org/record/4591458/files/nptel-train.tar.gz.partai
wget https://zenodo.org/record/4591466/files/nptel-train.tar.gz.partaj
wget https://zenodo.org/record/4593165/files/nptel-train.tar.gz.partak
wget https://zenodo.org/record/4593170/files/nptel-train.tar.gz.partal
wget https://zenodo.org/record/4593172/files/nptel-train.tar.gz.partam
wget https://zenodo.org/record/4593180/files/nptel-train.tar.gz.partan
wget https://zenodo.org/record/4593184/files/nptel-train.tar.gz.partao
wget https://zenodo.org/record/4593186/files/nptel-train.tar.gz.partap
wget https://zenodo.org/record/4593188/files/nptel-train.tar.gz.partaq
wget https://zenodo.org/record/4593192/files/nptel-train.tar.gz.partar
wget https://zenodo.org/record/4593195/files/nptel-train.tar.gz.partas
wget https://zenodo.org/record/4593199/files/nptel-train.tar.gz.partat
```

> If any download drops mid-way, resume with `wget -c <url>` instead of restarting.

---

## Step 6 — Reassemble & Extract
```bash
cat nptel-train.tar.gz.part* > nptel-train.tar.gz
mkdir -p ./train
tar -xzf nptel-train.tar.gz -C ./train
```

Clean up parts after extraction to recover space:
```bash
rm nptel-train.tar.gz.part*
rm nptel-train.tar.gz
```

---

## Step 7 — Download Pure Set

The Pure Set is a 1k manually annotated subset used for WER benchmarking in EDA Section 3:
```bash
wget https://github.com/AI4Bharat/NPTEL2020-Indian-English-Speech-Dataset/releases/download/v0.1/nptel-pure-set.tar.gz
mkdir -p ./nptel-pure-set
tar -xzf nptel-pure-set.tar.gz -C ./nptel-pure-set
```

---

## Step 8 — Launch EDA Notebook
```bash
uv run jupyter lab data/eda/english/NPTEL2020_EDA.ipynb
```

In Cursor, open the notebook and select the `global_notebooker` kernel.

Update `DATA_ROOT` and `SAMPLE_ROOT` in the notebook config cell:
```python
DATA_ROOT   = Path("/mnt/d/nptel2020/train")
SAMPLE_ROOT = Path("/mnt/d/nptel2020/nptel-pure-set")
```

---

## Directory Structure (Post-Setup)
```
/mnt/d/nptel2020/
├── train/                        # Extracted train split (~200 GB)
│   ├── <speaker_id>/
│   │   └── <chapter_id>/
│   │       ├── <chunk_id>.wav
│   │       ├── <chunk_id>.txt
│   │       └── <chunk_id>.json
├── nptel-pure-set/               # 1k manually annotated chunks
│   ├── <chunk_id>.wav
│   ├── <chunk_id>.txt
│   └── <chunk_id>_manual.txt
└── NPTEL2020_EDA.ipynb           # EDA notebook
```

---

## Environment Summary

| Item | Value |
|---|---|
| OS | Windows 11 + WSL2 |
| WSL Distribution | Ubuntu |
| Data Drive | `/mnt/d/nptel2020` |
| Free Space | 1.9 TB |
| Python Install Method | `pip --break-system-packages` |
| `librosa` Source | pip (unavailable via apt) |
| Dataset Format | LibriSpeech (`.wav` + `.txt` + `.json`) |
| Train Split Size (approx) | ~200 GB extracted |
