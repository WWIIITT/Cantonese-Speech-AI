# Cantonese Speech AI Research Framework

This repository is a notebook-first research workspace for Cantonese speech
recognition, dialect adaptation, tone/prosody modeling, and multimodal
speech-text understanding. The current starting point is `test.ipynb`, which
can be expanded into a sequence of experiment notebooks as datasets and models
are selected.

The goal is to build a practical research framework for low-resource Cantonese
speech AI: start with strong self-supervised speech models, add Cantonese tone
and dialect knowledge, then extend the system toward multimodal emotion-text
alignment and weakly supervised learning.

## System Framework

```text
Raw data
  -> Cantonese speech, transcripts, text corpora, emotion labels, noise samples
  -> data validation, normalization, segmentation, augmentation

Speech recognition
  -> SSL encoder: wav2vec 2.0, HuBERT, WavLM, Whisper, or MMS
  -> ASR decoder: CTC, seq2seq, or Whisper-style decoder
  -> language model rescoring and domain adaptation

Tone and linguistic context
  -> Jyutping/tone labels, F0 contours, duration, energy, prosody features
  -> tone-aware auxiliary loss and pronunciation-aware decoding

Dialect adaptation
  -> accent embeddings, sparse adapters, LoRA/adapters, soft-label transfer
  -> rapid adaptation using small dialect/accent-specific datasets

Multimodal understanding
  -> speech emotion embeddings, acoustic prosody vectors, text semantics
  -> attention-based fusion for emotion-aware recognition and analysis

Weak supervision
  -> pseudo-label generation, confidence filtering, label cleaning
  -> adversarial domain alignment for noisy and dialectal data

Evaluation and analysis
  -> CER/WER, tone error rate, dialect breakdown, noisy-speech robustness
  -> emotion classification or alignment metrics for multimodal tasks
```

## Job Duties -> Methods

| Duty area | Research and engineering methods | Expected output |
| --- | --- | --- |
| Low-resource Cantonese ASR | Fine-tune self-supervised speech encoders such as wav2vec 2.0, HuBERT, WavLM, MMS, or Whisper. Use speed perturbation, noise augmentation, SpecAugment, and multi-dataset training. | Baseline Cantonese ASR model with CER/WER reports. |
| Tone modeling | Add auxiliary tone classification, extract F0/pitch contours, map transcripts to Jyutping where possible, and measure tone substitution errors. | Tone-aware ASR experiments and tone error analysis. |
| Dialect adaptation | Train sparse adapter or LoRA modules per dialect/accent, learn accent embeddings, and transfer soft labels from a stronger teacher model. | Rapid adaptation workflow for Hong Kong Cantonese, regional accents, and noisy domains. |
| Context-aware recognition | Add pronunciation lexicons, prosody features, domain text, contextual biasing, and language-model rescoring. | Improved recognition for noisy, dialectal, and domain-specific audio. |
| Multimodal emotion-text alignment | Encode speech emotion/prosody features and text semantics, then fuse them with cross-attention or gated attention. | Emotion-aware speech-text representation for downstream classification or analysis. |
| Weakly supervised multimodal fusion | Generate pseudo labels, remove low-confidence examples, compare teacher-student agreement, and use adversarial alignment across domains. | Larger usable training set with cleaner labels and better domain robustness. |
| R&D leadership | Use experiment tracking, reproducible notebooks, code review, shared baselines, mentoring sessions, and written experiment reports. | A collaborative research workflow that teammates can reproduce and extend. |

## Dataset Candidates

Use dataset licenses carefully. Cantonese data availability changes over time,
so every dataset should be rechecked before download, redistribution, or model
release.

| Dataset | Use case | Notes |
| --- | --- | --- |
| [Mozilla Common Voice zh-HK](https://commonvoice.mozilla.org/zh-HK/datasets) / [Mozilla Data Collective](https://mozilladatacollective.com/) | Public Cantonese ASR starting point | Useful for baseline experiments and notebook prototyping. Check the current release version, hours, validated split, and license terms before training. |
| [MDCC: A Multi-Domain Cantonese Corpus](https://arxiv.org/abs/2201.02419) | Hong Kong Cantonese ASR research | Described as a 73.6-hour corpus with read speech and transcripts. Use the paper and repository/access instructions to confirm availability and license. |
| CantoMap / Cantonese MapTask-style corpus | Controlled dialogue or task-oriented speech | Reported as a 12.8-hour Cantonese MapTask corpus in dataset discussions, but a stable public source should be verified before depending on it. Treat as a candidate, not an assumed download. |
| HKCanCor / Hong Kong Cantonese Corpus | Linguistic analysis and transcript normalization | Useful for Cantonese linguistic patterns, particles, and conversational text. Treat it as a text/linguistic resource unless audio access is separately confirmed. |
| [WenetSpeech-Yue](https://arxiv.org/abs/2509.03959) | Large-scale Cantonese ASR research | Described as a 21,800-hour Cantonese corpus. Verify access path, license, and release status before planning experiments around it. |

Additional sources to consider after the first baseline:

- In-house Cantonese recordings with consent and clear data governance.
- Public Hong Kong media/audio where speech, transcript, and redistribution
  rights are explicitly allowed.
- Synthetic or weak labels generated by teacher ASR models, filtered by
  confidence and manual spot checks.
- Noise datasets for augmentation, such as public environmental noise corpora,
  if their licenses permit research use.

## Notebook Workflow

Model training can be handled in `.ipynb` notebooks first. Keep notebooks small,
reproducible, and focused on one experiment type.

| Notebook | Purpose |
| --- | --- |
| `test.ipynb` | Current environment check and GPU sanity check. |
| `01_data_discovery.ipynb` | Record dataset links, licenses, metadata, audio duration, transcript format, and split strategy. |
| `02_asr_baseline.ipynb` | Fine-tune or evaluate a Whisper/wav2vec-style baseline and report CER/WER. |
| `03_tone_modeling.ipynb` | Build tone labels, extract pitch/F0 features, add tone-aware auxiliary objectives, and analyze tone errors. |
| `04_dialect_adaptation.ipynb` | Train sparse adapters or LoRA modules, test accent embeddings, and compare soft-label transfer methods. |
| `05_multimodal_fusion.ipynb` | Fuse acoustic emotion embeddings with text semantic embeddings using attention-based modules. |
| `06_error_analysis.ipynb` | Break down errors by tone, speaker, dialect/accent, noise condition, and domain. |

Recommended notebook conventions:

- Put random seeds, model names, dataset versions, and hardware details in the
  first cell.
- Save metrics to a simple table or experiment tracker after each run.
- Keep raw data paths configurable so notebooks work on local machines,
  notebooks, and cloud GPU environments.
- Export reusable preprocessing or evaluation code into Python modules only
  after the notebook version is stable.

## Model Training Methods

### 1. Cantonese ASR Baseline

Start with a strong pretrained model instead of training from scratch. Good
first baselines are Whisper fine-tuning for sequence-to-sequence ASR, or
wav2vec 2.0 / HuBERT / WavLM fine-tuning with a CTC head.

Core steps:

1. Normalize audio to a consistent sample rate, such as 16 kHz.
2. Normalize transcripts: full-width/half-width characters, punctuation,
   Cantonese-specific particles, numerals, English mixing, and optional
   traditional/simplified policy.
3. Split by speaker or source to avoid leakage.
4. Train a baseline and report CER, WER if word segmentation is defined, and
   inference speed.
5. Add noise and reverberation augmentation after the clean baseline is known.

### 2. Tone and Prosody Modeling

Cantonese recognition errors often come from tone and near-homophone confusion.
Add tone-aware modeling as an auxiliary task rather than forcing the ASR decoder
to solve tone implicitly.

Methods:

- Convert transcript syllables to Jyutping with tone numbers where reliable.
- Extract F0, energy, duration, and voiced/unvoiced features with tools such as
  librosa, torchaudio, Praat/parselmouth, or pyworld.
- Add an auxiliary tone classifier on encoder states.
- Report tone error rate and confusion matrices in addition to CER/WER.
- Compare text-only decoding against prosody-aware rescoring.

### 3. Dialect and Accent Adaptation

Use parameter-efficient adaptation to avoid retraining the full ASR model for
every accent or recording condition.

Methods:

- Add sparse adapters, LoRA, or bottleneck modules to the encoder.
- Learn accent/domain embeddings from speaker region, source, or clustering.
- Use a teacher model to generate soft labels on unlabeled dialectal audio.
- Train student adapters on a mix of gold labels and confidence-filtered soft
  labels.
- Evaluate by dialect/accent subset, not only by global CER.

### 4. Context-Aware Recognition

Improve recognition in noisy and dialectal scenarios by injecting context from
linguistic features and domain text.

Methods:

- Build a Cantonese domain lexicon for names, places, products, and technical
  terms.
- Use language-model rescoring for n-best hypotheses.
- Add contextual biasing phrases for known topics.
- Include prosody and pause features for disambiguation when useful.
- Track performance separately for clean speech, noisy speech, code-switching,
  and domain-specific terms.

### 5. Multimodal Emotion-Text Alignment

For multimodal tasks, combine speech emotion signals with text semantics instead
of treating ASR text as the only representation.

Methods:

- Extract acoustic emotion features from prosody, speaker embeddings, or a
  pretrained speech emotion model.
- Encode ASR transcripts using a text encoder such as BERT-style Chinese models
  or multilingual sentence embeddings.
- Fuse acoustic and text vectors with cross-attention, gated attention, or
  late-fusion classifiers.
- Use contrastive or alignment losses when paired speech/text examples are
  available.
- Evaluate both recognition quality and downstream emotion or intent metrics.

### 6. Weakly Supervised Fusion

Weak supervision helps scale training beyond small labeled Cantonese datasets,
but pseudo labels must be cleaned carefully.

Methods:

- Generate pseudo transcripts and emotion labels with teacher models.
- Filter by ASR confidence, entropy, agreement between teachers, and audio
  quality checks.
- Manually inspect a small sample of accepted and rejected pseudo labels.
- Use adversarial domain alignment to reduce mismatch between clean labeled
  data and noisy weakly labeled data.
- Retrain in stages: clean supervised baseline, pseudo-label expansion, then
  domain-adapted fine-tuning.

## Evaluation Metrics

Minimum ASR metrics:

- Character error rate (CER)
- Word error rate (WER), if a stable Cantonese word segmentation policy exists
- Sentence error rate for strict command or keyword scenarios
- Real-time factor or latency for deployment planning

Tone and dialect metrics:

- Tone error rate
- Tone confusion matrix
- CER/WER by speaker, dialect/accent, source, and noise condition
- Code-switching error rate for Cantonese-English mixed speech

Multimodal metrics:

- Emotion classification F1 or macro-F1
- Text-speech alignment accuracy or retrieval metrics when applicable
- Domain-adversarial validation gap between labeled and weakly labeled data

## R&D Workflow

Recommended team process:

1. Keep one shared baseline notebook and one shared evaluation script.
2. Require every new method to compare against the same data split.
3. Log dataset version, model checkpoint, hyperparameters, metrics, and error
   samples for every experiment.
4. Review notebooks before promoting code into reusable modules.
5. Hold short research reviews for failed experiments, not only successful
   results.
6. Mentor new team members through reproducible baselines, dataset ethics, and
   model evaluation before assigning open-ended research tasks.

## Suggested Initial Milestones

| Milestone | Deliverable |
| --- | --- |
| M1: Data audit | Confirm usable Cantonese datasets, licenses, audio hours, transcript format, and train/dev/test splits. |
| M2: ASR baseline | Run a pretrained ASR baseline and report CER/WER on a small validated subset. |
| M3: Tone analysis | Add Jyutping/tone processing and produce tone error reports. |
| M4: Dialect adaptation | Compare full fine-tuning with adapter/LoRA adaptation on a dialect or noisy subset. |
| M5: Multimodal prototype | Build an attention-based speech emotion + text semantic fusion notebook. |
| M6: Weak supervision | Add pseudo-label cleaning and compare clean-only vs weakly supervised training. |

## Current Repository State

- `test.ipynb` contains a GPU environment check.
- `README.md` defines the planned research framework and notebook roadmap.
- Future notebooks and reusable modules should be added incrementally after the
  first dataset audit.
