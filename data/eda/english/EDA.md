---
title: "NPTEL2020 EDA — Environment & Data Setup Guide"
subtitle: "Project: Automated Correction of Speech Recordings (Low-Bandwidth Environments)"
author: "Akoua"
date: "`r Sys.Date()`"
output:
  html_document:
    toc: true
    toc_float: true
    theme: flatly
    highlight: tango
  pdf_document:
    toc: true
---
```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE, eval = FALSE)
```

---

## Required R Packages

Install these before knitting the document:
```{r install-packages}
install.packages(c(
  "knitr",       # RMarkdown rendering
  "rmarkdown",   # Document generation
  "kableExtra",  # Styled tables
  "reticulate"   # Run Python from R (optional, for inline Python chunks)
))
```

> **Note:** This document uses `bash` chunks to document shell commands.
> Set `eval=TRUE` on any chunk you want to actually execute while knitting,
> or run commands manually in your WSL terminal.

---

## Step 1 — Check Disk Space
```{bash check-disk}
df -h | grep -v tmpfs
```

**Result:** `D:\` had 1.9 TB free → chosen as data directory.

---

## Step 2 — Create Data Directory
```{bash make-dir}
mkdir -p /mnt/d/nptel2020
cd /mnt/d/nptel2020
```

---

## Step 3 — Install System Packages
```{bash system-packages}
sudo apt-get update -qq
sudo apt-get install -y ffmpeg libsndfile1 python3-pip python3-venv build-essential wget
sudo apt install python3-soundfile
```

---

## Step 4 — Install Python Packages

`python3-librosa` is unavailable via `apt` on this Ubuntu version.
All Python packages installed system-wide using the `--break-system-packages` flag:
```{bash pip-packages}
pip install \
  jiwer \
  pydub \
  nltk \
  jupyter \
  ipykernel \
  ipywidgets \
  numpy \
  pandas \
  matplotlib \
  seaborn \
  tqdm \
  librosa \
  --break-system-packages
```

### Python Package Reference
```{r package-table, echo=FALSE, eval=TRUE}
library(knitr)
library(kableExtra)

packages <- data.frame(
  Package = c(
    "librosa", "soundfile", "jiwer", "pydub", "nltk",
    "jupyter", "ipykernel", "ipywidgets",
    "numpy", "pandas", "matplotlib", "seaborn", "tqdm"
  ),
  Source = c(
    "pip", "apt + pip", "pip", "pip", "pip",
    "pip", "pip", "pip",
    "pip", "pip", "pip", "pip", "pip"
  ),
  Purpose = c(
    "Audio loading, mel spectrograms, energy features",
    "Fast WAV header reads without loading PCM",
    "Word Error Rate (WER) and CER computation",
    "Audio format conversion (WAV <-> OPUS via ffmpeg)",
    "OOV vocabulary analysis (words corpus)",
    "Notebook server",
    "Python kernel for Jupyter",
    "Interactive notebook widgets",
    "Numerical computing",
    "Tabular data manipulation",
    "Plotting and visualization",
    "Statistical visualizations",
    "Progress bars for loops"
  )
)

kable(packages, align = c("l", "c", "l")) %>%
  kable_styling(
    bootstrap_options = c("striped", "hover", "condensed"),
    full_width = TRUE
  ) %>%
  column_spec(1, bold = TRUE, monospace = TRUE) %>%
  column_spec(2, width = "6em") %>%
  row_spec(0, bold = TRUE, background = "#2E75B6", color = "white")
```

---

## Step 5 — Download Dataset (Train Split)

Data is hosted across 20 Zenodo records as split `.tar.gz` parts.
Run all commands from `/mnt/d/nptel2020`.
```{bash download-parts}
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
```{bash reassemble}
cat nptel-train.tar.gz.part* > nptel-train.tar.gz
mkdir -p ./train
tar -xzf nptel-train.tar.gz -C ./train
```

Optionally clean up parts after extraction to recover space:
```{bash cleanup}
rm nptel-train.tar.gz.part*
rm nptel-train.tar.gz
```

---

## Step 7 — Download Pure Set

The Pure Set is a 1k manually annotated subset used for WER benchmarking in EDA Section 3:
```{bash pure-set}
wget https://github.com/AI4Bharat/NPTEL2020-Indian-English-Speech-Dataset/releases/download/v0.1/nptel-pure-set.tar.gz
mkdir -p ./nptel-pure-set
tar -xzf nptel-pure-set.tar.gz -C ./nptel-pure-set
```

---

## Step 8 — Launch EDA Notebook
```{bash launch-notebook}
jupyter notebook /path/to/NPTEL2020_EDA.ipynb
```

> Update `DATA_ROOT` and `SAMPLE_ROOT` in the notebook config cell:
>
> ```python
> DATA_ROOT   = Path("/mnt/d/nptel2020/train")
> SAMPLE_ROOT = Path("/mnt/d/nptel2020/nptel-pure-set")
> ```

---

## Environment Summary
```{r env-summary, echo=FALSE, eval=TRUE}
library(knitr)
library(kableExtra)

env <- data.frame(
  Item = c(
    "OS",
    "WSL Distribution",
    "Data Drive",
    "Free Space",
    "Python Install Method",
    "librosa Source",
    "Dataset Format",
    "Train Split Size (approx)"
  ),
  Value = c(
    "Windows 11 + WSL2",
    "Ubuntu",
    "/mnt/d/nptel2020",
    "1.9 TB",
    "pip --break-system-packages",
    "pip (unavailable via apt)",
    "LibriSpeech (.wav + .txt + .json)",
    "~200 GB extracted"
  )
)

kable(env, col.names = c("Item", "Value"), align = c("l", "l")) %>%
  kable_styling(
    bootstrap_options = c("striped", "hover", "condensed"),
    full_width = TRUE
  ) %>%
  column_spec(1, bold = TRUE, width = "14em") %>%
  row_spec(0, bold = TRUE, background = "#2E75B6", color = "white")
```

---

## Directory Structure (Post-Setup)
```
/mnt/d/nptel2020/
├── train/                  # Extracted train split (~200 GB)
│   ├── <speaker_id>/
│   │   └── <chapter_id>/
│   │       ├── <chunk_id>.wav
│   │       ├── <chunk_id>.txt
│   │       └── <chunk_id>.json
├── nptel-pure-set/         # 1k manually annotated chunks
│   ├── <chunk_id>.wav
│   ├── <chunk_id>.txt
│   └── <chunk_id>_manual.txt
└── NPTEL2020_EDA.ipynb     # EDA notebook
```

---

## Notes & Troubleshooting

| Issue | Fix |
|---|---|
| `Unable to locate package python3-librosa` | Use `pip install librosa --break-system-packages` |
| `externally-managed-environment` pip error | Add `--break-system-packages` flag |
| `apt` can't find jiwer / pydub / nltk etc. | These are pip-only packages, not in apt |
| wget download interrupted | Re-run with `wget -c <url>` to resume |
| Not enough space on C:\ | Use D:\ via `/mnt/d/` — 1.9 TB available |
| Notebook can't find data | Update `DATA_ROOT` path in notebook cell 0 |