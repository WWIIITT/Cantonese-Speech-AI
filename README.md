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

For code-test preparation based on this project and the job duties, see
[`docs/interview_preparation.md`](docs/interview_preparation.md).
For focused code-test questions and model answers, see
[`docs/code_test_questions_answers.md`](docs/code_test_questions_answers.md).
For a repository-independent guide based directly on the advertised duties, see
[`docs/job_duties_code_test_qa.md`](docs/job_duties_code_test_qa.md).

## Setup

Use a virtual environment for notebook work so Jupyter uses the same packages
as this project.

### Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name cantonese-speech-ai --display-name "Cantonese Speech AI"
```

Then select the `Cantonese Speech AI` kernel in JupyterLab or VS Code.

Start local JupyterLab from the project virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
jupyter lab
```

Local paths can be configured by copying `.env.example` to `.env`. The `.env`
file is intentionally ignored by git.

For the first ASR phase, use only the local Common Voice Yue dataset:

```text
COMMON_VOICE_YUE_ROOT=Mozilla-HK-Speech-datasets/cv-corpus-26.0-2026-06-12/yue
VECTORENGINE_API_KEY=your_api_key_here
VECTORENGINE_BASE_URL=https://api.vectorengine.cn/v1
WHISPER_MODEL=whisper-1
```

### Local NVIDIA GPU

JupyterLab uses the GPU only when the selected notebook kernel points to a
Python environment with CUDA-enabled PyTorch installed.

First confirm Windows can see the GPU:

```powershell
nvidia-smi
```

If PyTorch reports a CPU build, such as `2.12.1+cpu`, reinstall PyTorch inside
this project's `.venv` with the CUDA wheel:

```powershell
.\.venv\Scripts\Activate.ps1
pip install --upgrade --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m ipykernel install --user --name cantonese-speech-ai --display-name "Cantonese Speech AI"
```

If you are setting up the project from scratch on a Windows machine with an
NVIDIA GPU, the full command sequence is:

```powershell
cd D:\GitHub\Cantonese-Speech-AI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install --upgrade --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m ipykernel install --user --name cantonese-speech-ai --display-name "Cantonese Speech AI"
jupyter lab
```

Restart JupyterLab, select the `Cantonese Speech AI` kernel, and run:

```python
import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
```

Expected on the tested local machine:

```text
2.11.0+cu128
12.8
True
NVIDIA GeForce RTX 4060 Laptop GPU
```

### Google Colab

Colab cannot read local Windows paths such as `D:\GitHub\...`. To run the
notebooks in Colab, copy the whole project folder to Google Drive:

```text
/content/drive/MyDrive/Cantonese-Speech-AI
```

The folder must contain both:

```text
src/cantonese_speech_ai/
Mozilla-HK-Speech-datasets/cv-corpus-26.0-2026-06-12/yue/
```

The first notebook cell will try to mount Google Drive and find this folder.
If you use a different Drive path, add this before the first import cell:

```python
import os
os.environ["PROJECT_ROOT"] = "/content/drive/MyDrive/your-folder-name"
```

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── test.ipynb
├── notebooks/
│   ├── 01_data_discovery.ipynb
│   ├── 02_asr_baseline.ipynb
│   ├── 03_tone_modeling.ipynb
│   ├── 04_dialect_adaptation.ipynb
│   ├── 05_multimodal_fusion.ipynb
│   └── 06_error_analysis.ipynb
├── src/cantonese_speech_ai/
│   ├── audio.py
│   ├── common_voice.py
│   ├── config.py
│   ├── datasets.py
│   ├── metrics.py
│   └── text.py
└── tests/
    ├── test_common_voice.py
    └── test_helpers.py
```

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

## Dataset Scope

The first implementation phase uses only Mozilla Common Voice Yue from the
local dataset folder. Keep the downloaded audio and TSV files out of git; the
folder is ignored by `.gitignore`.

PowerShell may display Cantonese TSV text as mojibake, but Python reads the
files correctly with `encoding="utf-8"`.

| Dataset | Use case | Notes |
| --- | --- | --- |
| Mozilla Common Voice Yue `cv-corpus-26.0-2026-06-12/yue` | Primary ASR dataset | Use `train.tsv`, `dev.tsv`, `test.tsv`, `validated.tsv`, `clip_durations.tsv`, and `clips/*.mp3`. Local counts are train 7420, dev 5130, test 5130, validated 191477. |
| [MDCC](https://arxiv.org/abs/2201.02419) | Future ASR expansion | Do not mix into the first baseline. Verify access and license later. |
| HKCanCor / Hong Kong Cantonese Corpus | Future linguistic support | Treat as text/linguistic reference unless audio access is confirmed. |
| [WenetSpeech-Yue](https://arxiv.org/abs/2509.03959) | Future large-scale ASR expansion | Verify release status, access path, and license before use. |

## Notebook Workflow

Model training can be handled in `.ipynb` notebooks first. Keep notebooks small,
reproducible, and focused on one experiment type.

| Notebook | Purpose |
| --- | --- |
| `test.ipynb` | Current environment check and GPU sanity check. |
| `notebooks/01_data_discovery.ipynb` | Load Common Voice Yue TSVs, join clip durations, check missing audio, and inspect metadata distributions. |
| `notebooks/02_asr_baseline.ipynb` | Run a small Whisper API zero-shot baseline on Common Voice dev audio, calculate CER/WER, and save prediction CSVs. |
| `notebooks/03_tone_modeling.ipynb` | Second-stage tone/Jyutping/prosody work after the ASR baseline is stable. |
| `notebooks/04_dialect_adaptation.ipynb` | Second-stage accent adaptation using the Common Voice `accents` column. |
| `notebooks/05_multimodal_fusion.ipynb` | Later-stage emotion-text fusion; Common Voice Yue has no native emotion labels. |
| `notebooks/06_error_analysis.ipynb` | Load prediction CSVs and analyze CER/WER by accent, age, gender, duration, and sentence length. |

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

Start with a strong pretrained model instead of training from scratch. The
first runnable baseline is Whisper API zero-shot transcription through the
VectorEngine-compatible OpenAI API. This produces a prediction CSV for error
analysis before local fine-tuning begins.

Core steps:

1. Normalize audio to a consistent sample rate, such as 16 kHz.
2. Normalize transcripts: full-width/half-width characters, punctuation,
   Cantonese-specific particles, numerals, English mixing, and optional
   traditional/simplified policy.
3. Split by speaker or source to avoid leakage.
4. Run Whisper API on a small `dev` subset first, such as 3, 20, then 100
   clips, and save `outputs/predictions/whisper_api_dev_<N>.csv`.
5. Report CER, WER if word segmentation is defined, and inference speed.
6. Move to local fine-tuning with Whisper, wav2vec 2.0, HuBERT, WavLM, or MMS
   after the zero-shot baseline and error analysis are stable.
7. Add noise and reverberation augmentation after the clean baseline is known.

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
- `notebooks/` contains starter experiment notebooks.
- `src/cantonese_speech_ai/` contains concise helper modules for config,
  Common Voice TSV loading, datasets, audio, text normalization, and metrics.
- `tests/` contains lightweight helper tests.
- `Mozilla-HK-Speech-datasets/` is the ignored local Common Voice Yue dataset.

## Verification

Run these checks after changing helper code or notebooks:

```powershell
python -m unittest discover -s tests -v
python -m compileall src
```
