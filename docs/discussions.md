# Discussions

A running log of interesting concepts and explanations.

---

## Time Domain vs. Other Kinds of Audio Inpainting
*2026-03-07*

Audio inpainting = reconstructing missing or corrupted segments of an audio signal. The "domain" refers to **where the reconstruction happens**.

### Time Domain
You work directly on the raw waveform samples (amplitude vs. time).

- **Input/output**: raw PCM samples (e.g., a float array at 16kHz)
- **Methods**: autoregressive models (WaveNet-style), sparse decomposition, polynomial interpolation
- **Strengths**: no information loss from transforms; natural for short gaps (< ~10ms)
- **Weaknesses**: very hard to model long-range structure; phase coherence must be handled implicitly

### Frequency / Spectral Domain
You work on the Short-Time Fourier Transform (STFT) — magnitude and/or phase spectrogram.

- **Input/output**: complex or magnitude spectrogram patches
- **Methods**: NMF-based methods, inpainting on spectrograms treated as 2D images (U-Net, etc.)
- **Strengths**: humans perceive audio spectrally; easier to model harmonic/tonal structure
- **Weaknesses**: phase reconstruction is non-trivial (Griffin-Lim or neural vocoders needed); STFT introduces time-frequency tradeoffs

### Learned Latent / Codec Domain
You work in the compressed latent space of a learned audio codec (EnCodec, DAC, SoundStream).

- **Input/output**: discrete tokens or continuous latent vectors
- **Methods**: masked token prediction (like BERT/MAR), diffusion in latent space
- **Strengths**: highly compressed representation captures perceptually meaningful structure; enables long-context modeling cheaply; state-of-the-art quality
- **Weaknesses**: quality bounded by codec reconstruction fidelity; requires a pre-trained codec

### Comparison Table

| | Time Domain | Spectral Domain | Latent/Codec Domain |
|---|---|---|---|
| Representation | raw samples | STFT / mel | codec tokens/vectors |
| Gap length | short (< 50ms) | medium (< 500ms) | long (seconds) |
| Phase handling | implicit | explicit problem | handled by codec |
| Compute | high (long sequences) | medium | low (compressed) |
| State-of-the-art? | rarely | partially | yes (2023–present) |

### Relevance to This Project

For **speech inpainting on customer call audio** (telephony, 8–16kHz, gaps likely from packet loss or corruption spanning 50ms–several seconds), the practical choice is almost certainly **latent/codec domain** — models like VoiceCraft, AudioSeal, or MAR-based approaches operate there and handle the gap lengths and speaker consistency you'd need. Pure time-domain methods only make sense for very short glitches.

---

## Music Inpainting vs. Speech Inpainting
*2026-03-07*

### Structural Differences

| | Speech | Music |
|---|---|---|
| **Signal type** | single voice, quasi-periodic | multiple instruments, complex mixtures |
| **Temporal structure** | phonemes (10–100ms), words, sentences | beats, bars, phrases (highly periodic) |
| **Frequency content** | 80Hz–8kHz, sparse harmonics | 20Hz–20kHz, dense harmonics |
| **Dynamics** | relatively uniform within an utterance | wide dynamic range, intentional silence |
| **Global structure** | local coherence (words follow words) | hierarchical (motif → phrase → section) |

### What's Easier in Speech

- **Single source**: one speaker, no mixing problem. The model only needs to reconstruct one voice, not disentangle instruments.
- **Constrained prosody**: speech has tighter rules — intonation contours, rhythm, and phoneme transitions are linguistically constrained. A language model prior (or even a phoneme-level prior) strongly guides reconstruction.
- **Shorter gap tolerance**: listeners accept slightly imperfect speech reconstruction more readily than musicians accept an imperfect note — speech intelligibility is the bar, not perceptual perfection.
- **More data**: massive speech corpora exist (LibriSpeech, Common Voice, VoxPopuli). Music datasets with stems/annotations are scarcer and often proprietary.
- **Simpler evaluation**: WER, MOS, PESQ, STOI are well-established. Music evaluation is much more subjective.

### What's Harder in Speech

- **Speaker identity**: the reconstructed segment must sound like the *same person* — timbre, accent, speaking rate, even emotional tone must be preserved. Music doesn't have speaker identity in the same way.
- **Linguistic content**: if the gap covers a word or phrase, the model must infer *what was said*, not just how it sounded. This conflates acoustic reconstruction with language modeling.
- **Telephony degradation**: real-world speech inpainting has to deal with codec artifacts, background noise, and channel distortion *on top of* the gap — the reference signal itself is degraded.
- **Short-context dependency**: a gap of even 100ms in speech can correspond to a full phoneme or consonant cluster. The model needs very fine-grained temporal precision.

### What's Easier in Music

- **Periodicity**: music is rhythmically structured. Beat tracking gives you a strong prior on *when* notes start and end, making gap boundaries much more predictable.
- **Instrument separation**: for stems-based inpainting (e.g., a piano track with a gap), you're working on a single instrument in isolation — simpler than full-mix reconstruction.
- **Tolerance for ambiguity**: if a guitar chord is missing, many voicings are musically plausible. The listener may not notice an alternative that fits the harmonic context.

### What's Harder in Music

- **Polyphony**: multiple simultaneous notes/instruments must all be reconstructed coherently — missing one creates audible harmonic clashes.
- **Long-range structure**: a gap in bar 8 must be consistent with bar 4 (motif repetition) and bar 16 (resolution). Music has hierarchical dependencies over much longer timescales than speech.
- **Timbre precision**: a missing violin note must match the exact timbre of the recorded violin in that recording — not just any violin. Timbre is highly sensitive to room acoustics, bow pressure, etc.
- **Rhythmic alignment**: reconstructed content must land precisely on the beat grid. A few milliseconds of drift is immediately noticeable in music; in speech it rarely matters.
- **Scarce annotated data**: music inpainting research is limited by the lack of large, clean, multi-track datasets with ground-truth gaps.

### Bottom Line for This Project

Speech inpainting is the right starting point. The problem is more constrained, better-resourced, and has clearer success criteria. The main challenge unique to the customer call setting is **speaker identity preservation under telephony conditions** — that's where most of the effort will go, and it has no real analog in music inpainting.
