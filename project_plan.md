# Project: Hindi Speech Inpainting for Recorded Customer Call Restoration

You are building a state-of-the-art Hindi speech inpainting system for restoring corrupted or partially lost segments in recorded customer calls.

This is **not** a real-time PLC system. It is a high-quality post-processing restoration pipeline for archived call recordings.

The system will be trained on the Vaani Hindi dataset with realistic telephony-style simulation and evaluated using signal-based and task-based metrics.

**Work through the sections below in order. Complete all tasks in a section before moving to the next.**

---

## 0. Global Setup

**Goal:** Create a reproducible, GPU-ready research environment with strong experiment tracking.

### Project Structure

```
src
configs
scripts
notebooks
data
  raw
  processed
  manifests
experiments
tests
docs
```

### Tasks

1. Use **uv** (preferred) or **poetry** for dependency management.
2. Install core dependencies:
   - PyTorch + CUDA
   - Lightning or Accelerate
   - torchaudio
   - transformers
   - datasets
   - librosa
   - soundfile
   - wandb or tensorboard
   - hydra-core
   - evaluate
   - pydantic
   - black, isort, ruff, pre-commit
3. Implement:
   - `src/config.py`
   - `src/utils/audio.py`
   - `src/utils/stft.py`
   - `src/utils/distributed.py`
4. Add CLI scripts:
   - `train_inpainting.py`
   - `eval_inpainting.py`
   - `prepare_vaani_hindi.py`
   - `preview_telephony_audio.py`
   - `visualize_masks.py`
5. Add pre-commit and CI:
   - formatting checks
   - unit tests on push
   - save git commit hash and full config per run

**Exit condition:** CLI help commands run without error.

---

## 1. Data Acquisition: Vaani Hindi

**Goal:** Download and index the Hindi portion of Vaani.

### Tasks

1. Use the Hugging Face dataset **ARTPARK-IISc/Vaani**.
2. Filter:
   - language equals Hindi
   - valid audio paths
   - reasonable duration bounds
3. Create speaker-based train, validation, and test splits.
4. Write manifest files in JSONL format containing: `audio_path`, `speaker_id`, `duration`, `language`, `split`.
5. Add unit tests:
   - No speaker overlap between splits
   - All language values equal `hi`
   - Files exist and are readable
6. Document dataset licensing and attribution in README.

**Exit condition:** Manifests are valid and sample audio is playable.

---

## 2. Telephony-Style Simulation

**Goal:** Simulate realistic recorded call artifacts.

Since this is recorded-call restoration, realism is more important than strict 8 kHz constraints.

Implement **`telephony_augment.py`** with:

- Resampling to 8 kHz and 16 kHz
- Bandpass filtering (300–3400 Hz)
- Mu-law companding
- Additive background noise (office, call-center, street)
- Random SNR between 0 and 20 dB
- Automatic gain control simulation
- Dynamic range compression
- Mild clipping simulation
- Optional codec simulation (G.711 or lightweight Opus)

Create **`configs/data/telephony.yaml`** to control probabilities and strengths.

**Exit condition:** Augmented samples sound like realistic Indian call recordings.

---

## 3. Synthetic Packet-Loss Modeling

**Goal:** Simulate realistic bursty corruption patterns.

Recorded calls often exhibit burst packet loss rather than uniform gaps.

Implement **`masking.py`** with:

1. Packet-based masking aligned to 20 ms.
2. Bursty loss model (Markov or geometric burst lengths).
3. Gap modes:
   - **short** (20–60 ms)
   - **medium** (60–200 ms)
   - **long** (200–400 ms)
4. Crossfade smoothing at boundaries.
5. Time-frequency mask generation.

Return `corrupted_input`, `mask`, and `clean_target`.

**Exit condition:** Visualizations show realistic burst patterns.

---

## 4. Strong Baselines

**Goal:** Establish solid reference points.

### Baselines

1. Zero-fill.
2. Copy-last-packet or waveform crossfade.
3. LPC interpolation.
4. Time-frequency U-Net.

### U-Net variants

- Magnitude-only prediction.
- Complex STFT prediction.
- Optional complex ratio mask.

### Loss

- Mask-weighted L1 or L2.
- Multi-resolution STFT loss.

### Reconstruction

- Inverse STFT.
- Optional phase refinement.

**Exit condition:** U-Net significantly outperforms classical baselines.

---

## 5. SSL-Based Inpainting

**Goal:** Use self-supervised speech encoders for latent-space inpainting.

### Steps

1. Select multilingual wav2vec 2.0 or HuBERT strong on Indic languages.
2. Align masks with encoder stride.
3. Extract latent features.
4. Apply latent-time masks.
5. Train inpainting decoder (Transformer or Conv1D).
6. Add lightweight vocoder (e.g., HiFi-GAN variant).

### Training modes

- Encoder frozen.
- Encoder fine-tuned.

### Losses

- Latent L1 or L2.
- Waveform STFT loss.
- Mask-weighted emphasis.

**Exit condition:** SSL model beats U-Net for gaps up to 300 ms.

---

## 6. Discrete Token Diffusion (Optional Phase 2)

*Proceed only if SSL model clearly improves metrics.*

**Goal:** Token-based diffusion inpainting for highest perceptual quality.

1. Integrate audio tokenizer (Encodec or SoundStream style).
2. Implement masked token diffusion.
3. Modify tokens only in masked regions.
4. Crossfade boundaries in waveform space.
5. Add sampling script.

**Exit condition:** Audible improvement over SSL model.

---

## 7. Evaluation Framework

### Objective metrics

- PESQ (optional)
- STOI
- SI-SDR
- Mask-region-only metrics
- Gap-length-specific metrics

### Task-based metric

Since this is customer-call restoration:

- Run Hindi ASR on clean, corrupted, and inpainted audio.
- Measure WER recovery relative to corrupted audio.

### Outputs

- CSV per file
- Aggregated JSON
- Gap-length plots
- Inference latency

**Exit condition:** Clear comparison tables across models.

---

## 8. Domain Adaptation to Real Recorded Calls

**Goal:** Bridge synthetic-to-real gap.

1. Add **`calls_india.py`** dataset loader.
2. Apply same telephony preprocessing.
3. Mix Vaani and real calls during fine-tuning.
4. Simulate corruption on real audio.
5. Re-run evaluation.

**Exit condition:** Improved robustness on real recordings.

---

## 9. Offline Processing Optimization

Since this is post-processing:

### Priorities

- Quality over ultra-low latency
- Batch processing
- GPU efficiency

### Tasks

- Profile inference time per 10-second file.
- Implement batched inference mode.
- Optional quantization.
- Export ONNX and TorchScript.
- Provide CLI tool:

  ```bash
  python restore_call.py --input corrupted.wav --output restored.wav
  ```

---

## 10. Documentation and Reproducibility

### Required

- README with setup instructions and experiment commands.
- `docs/report.md` with architecture diagrams and results.
- Hydra-based experiment logging.
- Save full configs and git commit per run.

### Ablations

- Classical vs U-Net
- Magnitude vs complex prediction
- SSL frozen vs fine-tuned
- Telephony augmentation on vs off
- Bursty vs uniform masking
- Domain adaptation impact

---

## Final Success Criteria

The system:

- Outperforms classical PLC methods.
- Improves PESQ, STOI, and SI-SDR across 20–400 ms gaps.
- Reduces ASR WER relative to corrupted audio.
- Generalizes to real recorded customer calls.
- Provides a clean, reproducible research pipeline.
- Supports batch offline restoration.
