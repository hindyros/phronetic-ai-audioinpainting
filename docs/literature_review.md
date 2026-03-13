# Top 10 State‑of‑the‑Art Papers for Speech / Audio Inpainting & Packet‑Loss Concealment

Below is a curated list of 10 leading‑edge papers directly relevant to **audio inpainting** and **packet‑loss concealment (PLC)** for your VoIP‑like Indian call‑center use case. Each entry includes a link, the core architecture / technique, and a brief summary of the reported results.

---

## 1. *Phoneme‑Guided Diffusion Inpainting (PGDI) for speech with long gaps*  
**Paper:** https://arxiv.org/html/2508.08890v1  
**Architecture & technique:**  
- Diffusion‑based speech inpainting using a **denoising diffusion model conditioned on phoneme‑level text**.  
- Uses an ASR + LM to extract a text transcript once per utterance, then uses this text to guide the diffusion generator for missing segments.  
- The model is trained to reconstruct large gaps (up to ≈1 second) while preserving speaker identity, prosody, and room acoustics.  

**Results:**  
- Outperforms prior non‑diffusion methods on objective metrics (PESQ, STOI, WER) and subjective listening tests.  
- Shows strong performance even when text is not available at inference, though quality further improves with transcript guidance.  

---

## 2. *Audio Inpainting using Discrete Diffusion Model (AIDD)*  
**Paper:** https://arxiv.org/html/2507.08333v1  
**Architecture & technique:**  
- **Discrete diffusion model** operating on audio tokens rather than raw waveforms or spectrograms.  
- Combines a **WavTokenizer**‑style encoder to quantize audio into a compact sequence of discrete tokens, then applies a **categorical diffusion model** over these tokens for inpainting.  
- Waveform is reconstructed by decoding the tokens.  

**Results:**  
- Achieves excellent perceptual quality on music and speech inpainting, even for challenging gap scenarios.  
- For ≈300 ms gaps, the model reports **FAD≈3.81**, lower than a prior continuous‑diffusion baseline (≈4.9), indicating more realistic reconstructions.  

---

## 3. *A Fullband Neural Network for Audio Packet Loss Concealment*  
**Paper:** https://ieeexplore.ieee.org/document/10627667 (IEEE Xplore PDF)  
**Architecture & technique:**  
- End‑to‑end **full‑band (48 kHz) neural‑network PLC** for real‑time VoIP.  
- Uses a deep, time‑domain (or TF‑domain) architecture that takes received packets around a gap and predicts the missing waveform.  
- Designed to satisfy strict latency constraints of the ICASSP 2024 PLC Challenge.  

**Results:**  
- Achieves **state‑of‑the‑art PLCMOS** scores on the ICASSP 2024 full‑band PLC challenge test set.  
- Outperforms classical codecs’ built‑in PLC and earlier deep PLC models in both P.804‑based listening tests and ASR‑based intelligibility (WER).  

---

## 4. *BS‑PLCNet 2: Two‑stage Band‑split Packet Loss Concealment Network*  
**Paper:** https://arxiv.org/abs/2406.05961  
**Architecture & technique:**  
- Two‑stage **band‑split PLC network** for full‑band audio (e.g., 48 kHz).  
- Splits the signal into low‑band and wide‑band paths; the wide‑band module uses a **dual‑path encoder** (non‑causal + causal) to exploit future context while training a causal student via **intra‑model knowledge distillation**.  
- Adds a lightweight **post‑processing module** after PLC to clean distortions and noise.  

**Results:**  
- With only **≈40% of BS‑PLCNet’s parameters**, BS‑PLCNet 2 improves **PLCMOS by 0.18** on the ICASSP 2024 blind evaluation set.  
- Sets new SOTA on the ICASSP 2024 challenge in terms of quality vs. complexity trade‑off.  

---

## 5. *The ICASSP 2024 Audio Deep Packet Loss Concealment Grand Challenge*  
**Paper:** https://ieeexplore.ieee.org/iel8/10625769/10625780/10626178.pdf (ICASSP 2024 challenge overview)  
**Architecture & technique:**  
- This is a **challenge paper** describing the dataset, evaluation procedure, and top systems (including winners).  
- Top systems use **encoder–decoder** or **band‑split causal networks** with careful buffering and latency constraints (≤20 ms lookahead).  

**Results:**  
- The challenge benchmarks 9 submitted systems under **P.804 MOS** and **WER**; the three top systems achieve significantly higher perceptual quality and intelligibility than legacy PLC methods.  
- Provides a clear reference for what architecture families and evaluation metrics matter for real‑time PLC.  

---

## 6. *A Time‑Domain Packet Loss Concealment Method by Designing CRN‑Trans*  
**Paper:** https://ieeexplore.ieee.org/document/10400230  
**Architecture & technique:**  
- **CRN‑Trans**: a time‑domain **Convolutional Recurrent Network with Transformer** for PLC.  
  - Convolutional layers extract local features per frame.  
  - Transformer + LSTM jointly model temporal dependencies; self‑attention focuses on relevant prior frames to reconstruct lost packets.  
- Designed for low‑latency real‑time use.  

**Results:**  
- Outperforms traditional PLC and some earlier deep‑network baselines on **PESQ** and **STOI** for short gaps (typical VoIP packet sizes).  
- Shows particular gains on noisy conditions, suggesting that combining attention with recurrence helps contextual PLC.  

---

## 7. *Janssen 2.0: Audio Inpainting in the Time‑Frequency Domain*  
**Paper:** https://eusipco2025.org/wp-content/uploads/pdfs/0000301.pdf (EUSIPCO 2025)  
**Architecture & technique:**  
- **Time‑frequency domain inpainting** applied to spectrogram gaps.  
- Uses a **deep prior‑inspired U‑Net‑like** network that predicts missing time‑frequency bins conditioned on surrounding bins.  
- The model is optimized for both speech and music inpainting.  

**Results:**  
- Beats the DPAI deep‑prior inpainting baseline on **objective** and **subjective tests**.  
- Shows particular strength when gaps are longer or more irregular, thanks to the TF‑domain modeling and skip‑connection design.  

---

## 8. *Is Self‑Supervised Learning Enough to Fill in the Gap? A Study on Speech Inpainting*  
**Paper:** https://www.sciencedirect.com/science/article/abs/pii/S0885230825001470 (ScienceDirect)  
**Architecture & technique:**  
- Explores whether **pre‑trained self‑supervised learning (SSL) encoders** (e.g., HuBERT‑style) can be used for speech inpainting **with little or no additional training**.  
- Uses a frozen SSL encoder and a simple decoder that predicts the full latent sequence from masked portions, then reconstructs audio.  

**Results:**  
- Shows that SSL‑based methods can already inpaint reasonably well without being trained from scratch for the inpainting task.  
- Quantifies the gap between SSL‑only and full end‑to‑end supervised models; argues that **some fine‑tuning is still beneficial** for high‑quality, long‑gap reconstruction.  

---

## 9. *Speech Inpainting: Context‑based Speech Synthesis Guided by Video*  
**Paper:** https://arxiv.org/abs/2306.00489  
**Architecture & technique:**  
- **Audio‑visual speech inpainting** where a corrupted audio segment is reconstructed using **visual face cues** as context.  
- Uses a **multi‑modal encoder** (audio + video) and a speech generator conditioned on visual and audio context.  

**Results:**  
- Achieves improved intelligibility and perceptual quality compared with audio‑only baselines on AV‑synchronized datasets.  
- Demonstrates that **visual context can significantly reduce uncertainty** in speech inpainting, especially for long gaps or noisy conditions.  

---

## 10. *Transonic: Packet‑Loss Concealment using Transformer Networks* (representative of recent Transformer‑PLC)  
*(If you want an explicit Transformer‑only PLC paper, this is a placeholder class; most recent SOTA variants like CRN‑Trans above absorb these ideas.)*  

**Typical architecture & technique:**  
- **Transformer‑based PLC** that models the audio sequence as a discrete token sequence or latent vector sequence.  
- Uses masked self‑attention to attend only to past and limited future frames, making it suitable for low‑latency PLC.  
- Often combined with convolutional or RNN front‑ends for local feature extraction.  

**Representative results across multiple papers:**  
- Transformers beat purely RNN‑based PLC on **PESQ/STOI** and **MOS**, especially for bursts of consecutive packet loss.  
- Recent work (e.g., CRN‑Trans and BS‑PLCNet 2) shows that **hybrid architectures** (Conv + RNN + Transformer) give the best balance of latency, robustness, and quality.

---

## 11. *LEMAS: A 150K‑Hour Large‑scale Extensible Multilingual Audio Suite with Generative Speech Models*
**Paper:** https://arxiv.org/abs/2601.04233
**Code / Dataset:** https://github.com/LEMAS-Project · https://huggingface.co/LEMAS-Project
**Architecture & technique:**
- Introduces **LEMAS-Dataset**: 150,000 hours of multilingual speech across 10 languages (Chinese, English, Russian, Spanish, Indonesian, German, Portuguese, Vietnamese, French, Italian) with rigorous **word‑level timestamps** obtained via MMS forced alignment — the largest open‑source multilingual corpus with temporal annotations.
- Trains two benchmark generative models on this dataset:
  - **LEMAS-TTS** — non-autoregressive **flow‑matching** TTS built on the F5-TTS architecture, using a Diffusion Transformer (DiT) backbone. Adds a **CTC alignment loss** (to enforce phoneme–acoustic monotonicity) and an **accent‑adversarial disentanglement** objective via a Gradient Reversal Layer, suppressing cross‑lingual accent leakage. A cross‑lingual prosody encoder (ECAPA-TDNN) provides fine‑grained prosody conditioning.
  - **LEMAS-Edit** — autoregressive **codec‑based speech editing** model extending VoiceCraft, which frames inpainting as a **masked token infilling** task. Introduces a history‑aware repetition penalty to suppress silence loops, an adaptive re‑generation mechanism that monitors speaking rate to prevent run‑away decoding, and a full signal‑enhancement pipeline (UVR5 for mild denoising, DeepFilterNet for aggressive suppression).
- The editing pipeline uses **Whisper large** for multilingual ASR and the **MMS Forced Aligner** for precise word‑level boundary detection, enabling surgical, word‑granularity inpainting with natural transitions.

**Results:**
- LEMAS‑TTS achieves lower WER and higher speaker similarity than OpenAudio‑S1‑Mini across all 10 languages (avg. WER **6.39 %** vs. 12.27 %).
- LEMAS‑Edit produces seamless, smooth‑boundary edits on in‑the‑wild noisy recordings, including audio with background noise comparable to call‑center conditions.
- Subjective A/B tests show LEMAS‑Edit is preferred for naturalness over LEMAS‑TTS on editing tasks across seven languages.

**Relevance to this project:**
- LEMAS‑Edit is architecturally the closest prior work to our goal: codec‑based masked infilling with precise temporal alignment applied to noisy, in‑the‑wild speech — directly analogous to our customer‑call restoration task.
- The MMS aligner used in their pipeline natively supports **Hindi** (it covers 1,100 + languages), so the alignment component transfers directly.
- **Hindi is not among the 10 LEMAS training languages**, but the open‑source codebase and warm‑start from the 330 M‑parameter VoiceCraft checkpoint provide a practical fine‑tuning path (see project plan Section 6).
- The signal‑enhancement components (DeepFilterNet, UVR5) directly address telephony noise present in our target recordings.

---

You can use these papers as references to motivate your choice of:
- **TF‑domain U‑Net + mask‑aware loss** (inspired by Janssen 2.0).
- **SSL‑based inpainting with HuBERT‑style encoders** (from *Is Self‑Supervised Learning Enough…*).
- **Diffusion‑based or discrete‑diffusion speech inpainting** for long gaps (PGDI and AIDD).
- **Band‑split or low‑latency Transformer‑based PLC** for real‑time deployment (BS‑PLCNet 2, CRN‑Trans, ICASSP 2024 challenge systems).
- **Autoregressive codec inpainting with temporal alignment** for word‑level editing on noisy speech (LEMAS‑Edit / VoiceCraft).  
