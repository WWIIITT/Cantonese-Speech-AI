# Cantonese Speech AI Code Test Interview Preparation

This guide prepares you for a one-hour coding assessment based on the Cantonese
speech AI job duties. The likely focus is not full model training, but whether
you can reason clearly about speech data, ASR metrics, low-resource modeling,
tone/dialect issues, and practical Python/Jupyter implementation.

## How To Present Your Project

Use this short introduction if they ask what you have prepared:

> I built a notebook-first Cantonese speech AI research scaffold using Mozilla
> Common Voice Yue. The workflow covers data discovery, Whisper API baseline
> prediction, CER/WER evaluation, Cantonese colloquial and tone-aware error
> analysis, dialect adaptation planning, and a multimodal fusion scaffold. The
> project is designed for low-resource Cantonese ASR, where pretrained speech
> models are combined with tone, prosody, accent, and weak-supervision methods.

If you need a shorter version:

> I start from a strong pretrained ASR baseline, analyze Cantonese-specific
> errors with CER and tone/colloquial markers, then design improvements using
> tone modeling, adapter-based dialect adaptation, and weakly supervised
> multimodal fusion.

## What They May Test

Expect tasks around these areas:

- Reading `.tsv` or `.csv` speech metadata with Pandas.
- Joining audio metadata such as duration with transcript rows.
- Computing CER or WER.
- Explaining why Cantonese ASR is difficult.
- Designing a low-resource ASR training pipeline.
- Explaining self-supervised speech models such as wav2vec 2.0, HuBERT, WavLM,
  or Whisper.
- Explaining tone modeling and Cantonese colloquial transcript errors.
- Designing dialect adaptation using adapters, sparse tuning, or soft labels.
- Explaining multimodal speech-text fusion.
- Writing clean notebook code under time pressure.

## Core Questions And Strong Answers

### 1. How would you build a Cantonese ASR system for low-resource data?

**Answer:**

I would avoid training from scratch. Cantonese labeled speech is limited, so I
would start with a pretrained self-supervised or large-scale speech model such
as Whisper, wav2vec 2.0, HuBERT, or WavLM. The first stage is a reproducible
data pipeline: read Common Voice Yue metadata, resolve audio paths, normalize
transcripts carefully, compute duration, and split train/dev/test.

Then I would build a baseline and evaluate with CER because Chinese text does
not have reliable whitespace word boundaries. After that, I would inspect error
patterns, especially Cantonese colloquial words such as `唔`, `嘅`, `咗`, `哋`,
and `嚟`. If the model converts them into written Chinese such as `不`, `的`,
`了`, `們`, or `來`, I would treat that as a style and recognition error for a
Cantonese transcript task.

For improvement, I would add data augmentation, tone-aware auxiliary training,
language model rescoring, and lightweight adaptation modules for accent or
speaker variation.

### 2. Why is Cantonese ASR harder than English ASR?

**Answer:**

Cantonese ASR is harder for several reasons:

- It is lower-resource than English, so there is less labeled speech.
- Cantonese has lexical tones, and tone changes can change word meaning.
- Written Chinese and spoken Cantonese differ strongly.
- Many ASR models output standard written Chinese instead of Cantonese-style
  transcripts.
- There are accent differences across Hong Kong, Guangdong, overseas speakers,
  and different recording conditions.
- Chinese text has no natural whitespace segmentation, so evaluation and
  language modeling need careful handling.

For example, a model may hear Cantonese correctly but output `我不知道` instead
of `我唔知道`. For a Cantonese ASR target, that is still an error because the
expected transcript preserves spoken Cantonese wording.

### 3. What is CER, and why use it for Cantonese?

**Answer:**

CER means Character Error Rate. It measures the edit distance between the
reference transcript and predicted transcript, divided by the number of
characters in the reference.

```text
CER = (Substitutions + Deletions + Insertions) / Reference character count
```

Example:

```text
Reference:  我唔知道
Prediction: 我不知道
```

Only one character differs: `唔` was changed to `不`. The reference has four
characters, so the CER is `1 / 4 = 0.25`.

CER is useful for Cantonese and Chinese ASR because word boundaries are not
explicit. WER depends on segmentation, while CER is more stable.

### 4. What is WER, and when would you use it?

**Answer:**

WER means Word Error Rate. It is similar to CER, but compares words instead of
characters.

```text
WER = (Word substitutions + Word deletions + Word insertions) / Reference word count
```

For English, WER is usually the main ASR metric because words are separated by
spaces. For Cantonese or Chinese, WER is less stable unless we define a
consistent segmentation method. I would report CER as the main metric and WER
as a secondary metric if segmentation is controlled.

### 5. What is self-supervised learning in speech?

**Answer:**

Self-supervised speech learning uses unlabeled audio to learn acoustic
representations. Instead of requiring transcripts for every audio file, the
model creates a pretext task from the signal itself.

Examples:

- wav2vec 2.0 masks speech representations and predicts quantized latent units.
- HuBERT predicts cluster assignments from masked speech frames.
- WavLM extends SSL speech learning with better speaker and noise robustness.

This is useful for Cantonese because we can use large unlabeled audio or
pretrained multilingual models, then fine-tune on a smaller Cantonese labeled
dataset.

### 6. How would you fine-tune wav2vec 2.0 or HuBERT for Cantonese ASR?

**Answer:**

I would prepare audio at the required sample rate, usually 16 kHz, normalize
transcripts, and build a character-level tokenizer or use an existing tokenizer
if appropriate. Then I would fine-tune the pretrained encoder with a CTC loss.

The training loop would include:

- Load audio and transcript pairs.
- Resample audio to 16 kHz.
- Tokenize transcript into character IDs.
- Use dynamic padding for variable-length audio.
- Train with CTC loss.
- Decode with greedy decoding or beam search.
- Evaluate CER on dev and test splits.

For low-resource Cantonese, I would start with a small subset to confirm the
pipeline, then scale up gradually.

### 7. What is CTC loss?

**Answer:**

CTC means Connectionist Temporal Classification. It is used when the input audio
sequence is much longer than the output transcript and we do not know the exact
alignment between audio frames and characters.

CTC allows the model to predict repeated tokens and blank tokens, then collapse
them into the final transcript. For example:

```text
Raw frame output: 我 _ 唔 唔 _ 知 知 道
Collapsed output: 我唔知道
```

This is common in speech recognition because frame-level labels are usually not
available.

### 8. How would you model Cantonese tones?

**Answer:**

I would treat tone modeling as an auxiliary task rather than replacing the ASR
objective. The main ASR model still predicts characters, while an additional
branch predicts tone-related labels.

Possible methods:

- Convert transcripts to Jyutping with tone numbers.
- Extract tone sequences such as `ngo5 m4 zi1 dou6`.
- Extract F0 or pitch contour from audio.
- Add an auxiliary tone classifier on top of encoder representations.
- Add a combined loss:

```text
total_loss = asr_ctc_loss + lambda_tone * tone_classification_loss
```

I would evaluate whether tone-aware training reduces character errors involving
tone-sensitive words.

### 9. What are limitations of automatic Jyutping conversion?

**Answer:**

Automatic Jyutping conversion is useful but imperfect. Cantonese has
polyphonic characters, context-dependent pronunciation, and colloquial forms.
If conversion is wrong, the tone label may also be wrong. Therefore, I would use
Jyutping-based tone features as exploratory labels or auxiliary supervision,
not as perfect ground truth.

In a production research setting, I would validate a subset manually or use a
high-quality pronunciation lexicon.

### 10. How would you handle Cantonese colloquial vs written Chinese errors?

**Answer:**

I would explicitly track colloquial marker errors. For example:

```text
嘅 -> 的
哋 -> 們
唔 -> 不
咗 -> 了
嚟 -> 來
```

If the reference uses spoken Cantonese but the prediction uses written Chinese,
the model may be semantically close but stylistically wrong for the dataset.
I would create an error taxonomy with categories such as:

- Low CER.
- High CER.
- Written-style replacement.
- Missing Cantonese colloquial marker.
- Possible tone/pronunciation confusion.

This helps decide whether to improve the acoustic model, the language model,
or the transcript normalization policy.

### 11. How would you build dialect-adaptive ASR?

**Answer:**

I would first analyze metadata such as accent, age, gender, duration, and error
rate. If some accent groups have higher CER, I would adapt the model using
lightweight methods instead of full fine-tuning.

Possible methods:

- Adapter layers inserted into the pretrained model.
- LoRA-style low-rank updates.
- Accent embeddings concatenated with encoder features.
- Sparse routing where only some adaptation parameters activate for each
  accent group.
- Teacher-student soft-label transfer from a stronger ASR model.

This reduces training cost and helps avoid overfitting small accent-specific
data.

### 12. What is sparse encoding for accent adaptation?

**Answer:**

Sparse encoding means the model does not update or activate all parameters for
every accent. Instead, it learns a compact accent representation or selects a
small subset of adapter parameters.

For example, a Hong Kong speaker and an overseas Cantonese speaker may share
the base ASR model but use different adapter activations. This allows fast
adaptation with fewer parameters.

### 13. What is soft-label transfer?

**Answer:**

Soft-label transfer is a teacher-student learning method. A strong teacher
model generates probability distributions, not only hard predictions. The
student model learns from these probabilities.

Hard label:

```text
Correct token = 唔
```

Soft label:

```text
唔: 0.72
不: 0.16
冇: 0.05
other: 0.07
```

Soft labels contain uncertainty and similarity information. This is useful when
training data is small or noisy.

### 14. How would you make ASR context-aware?

**Answer:**

I would add context at multiple levels:

- Acoustic context: duration, speaking rate, pitch, energy, and noise level.
- Linguistic context: Cantonese colloquial markers, domain vocabulary, language
  model rescoring.
- Metadata context: accent, age, gender, or recording source.
- Prompt context: domain hints for models such as Whisper.

For example, if the domain is Hong Kong daily conversation, the decoder should
prefer `唔`, `嘅`, and `咗` over written-style alternatives when the acoustic
evidence supports Cantonese colloquial speech.

### 15. How would you improve noisy-speech robustness?

**Answer:**

I would use both training-time and evaluation-time methods.

Training-time:

- Add background noise augmentation.
- Use speed perturbation.
- Use volume perturbation.
- Use SpecAugment.
- Mix clean and noisy data.

Evaluation-time:

- Report CER by noise condition.
- Report CER by duration bucket.
- Inspect short clips separately because short utterances can be unstable.
- Compare clean baseline against augmented model.

The goal is not only lower average CER, but more stable performance across
realistic recording conditions.

### 16. How would you design multimodal emotion-text alignment?

**Answer:**

I would build two encoders:

- A speech encoder for acoustic/emotion features such as pitch, energy,
  speaking rate, or embeddings from a pretrained speech emotion model.
- A text encoder for semantic features using BERT, RoBERTa, sentence
  embeddings, or an LLM embedding model.

Then I would fuse them with attention:

- Text-to-audio attention: text representation attends to emotional speech
  features.
- Audio-to-text attention: acoustic representation attends to semantic tokens.
- Cross-attention or co-attention for bidirectional fusion.

The fused representation can be used for emotion classification, intent
detection, or context-aware ASR correction.

### 17. What if Common Voice Yue has no emotion labels?

**Answer:**

Then I would not pretend it is an emotion dataset. I would build a scaffold and
use it for feature extraction and ASR error analysis first. For emotion fusion,
I would either add another labeled emotion dataset or create pseudo-labels from
a pretrained emotion classifier, then clean those labels by confidence.

This is important because forcing emotion training on a dataset without emotion
labels would produce weak or misleading results.

### 18. What is weak supervision?

**Answer:**

Weak supervision means the labels are incomplete, noisy, or automatically
generated. In speech-text multimodal learning, weak labels could come from:

- A pretrained ASR model.
- A pretrained emotion classifier.
- Heuristic rules.
- Metadata.
- Agreement between multiple models.

Weak supervision is useful when human annotation is expensive, but it requires
careful cleaning and confidence filtering.

### 19. How would you clean pseudo-labels?

**Answer:**

I would keep only high-confidence pseudo-labels at first. Then I would remove
samples where different models strongly disagree.

Possible cleaning rules:

- Keep predictions above a confidence threshold.
- Remove samples with very high ASR CER if transcript quality is bad.
- Use agreement between audio emotion model and text sentiment model.
- Down-weight uncertain samples instead of deleting all of them.
- Manually inspect a small validation subset.

This improves training stability and avoids teaching the student model wrong
patterns.

### 20. What is adversarial alignment?

**Answer:**

Adversarial alignment tries to make embeddings from different domains or
modalities similar. A discriminator tries to detect whether an embedding comes
from speech or text. The encoder tries to produce embeddings that confuse the
discriminator.

In multimodal speech-text learning, this can help align audio emotion features
with text semantic features, especially when labels are weak or noisy.

### 21. How would you evaluate this whole system?

**Answer:**

I would evaluate each part separately and together:

- ASR accuracy: CER as primary metric, WER as secondary metric.
- Tone behavior: tone sequence mismatch or tone-related error categories.
- Dialect robustness: CER by accent, age, gender, duration, and speaker group.
- Noisy robustness: CER under clean/noisy or augmented conditions.
- Multimodal quality: emotion classification F1, retrieval accuracy, or
  alignment similarity depending on the task.
- Error analysis: inspect worst examples, written-style replacements, and
  colloquial marker loss.

Good evaluation should show not only average performance, but where the model
fails.

### 22. How would you mentor an R&D team on this project?

**Answer:**

I would set up a reproducible workflow:

- Standard notebook order: data discovery, baseline, tone analysis, adaptation,
  multimodal scaffold, error analysis.
- Shared helper modules for repeated logic such as metrics, text normalization,
  and dataset loading.
- Clear experiment tracking with dataset version, model name, parameters, and
  metrics.
- Code review focused on reproducibility, leakage prevention, and correct
  evaluation.
- Short training sessions on ASR metrics, Cantonese-specific error patterns,
  and PyTorch/Jupyter workflow.

The goal is to help the team move quickly without losing scientific discipline.

## Likely Coding Tasks

### Task 1: Read Common Voice TSV

```python
from pathlib import Path
import pandas as pd

root = Path("Mozilla-HK-Speech-datasets/cv-corpus-26.0-2026-06-12/yue")
dev = pd.read_csv(root / "dev.tsv", sep="\t", encoding="utf-8")

print(dev.shape)
print(dev[["path", "sentence", "accents", "age", "gender"]].head())
```

What to explain:

- `sep="\t"` is required because Common Voice metadata is TSV.
- `encoding="utf-8"` preserves Cantonese text.
- Metadata can be used for dialect and error analysis.

### Task 2: Add Audio Paths And Duration

```python
durations = pd.read_csv(root / "clip_durations.tsv", sep="\t", encoding="utf-8")
durations = durations.rename(columns={"clip": "path"})

df = dev.merge(durations, on="path", how="left")
df["audio_path"] = df["path"].apply(lambda p: str(root / "clips" / p))
df["duration_sec"] = df["duration[ms]"] / 1000

df[["audio_path", "sentence", "duration_sec"]].head()
```

What to explain:

- Do not load all audio waveforms during metadata discovery.
- Join duration first because it is cheap and useful for filtering.
- Full waveform loading should happen only inside training or audio inspection
  steps.

### Task 3: Implement CER

```python
def cer(reference, hypothesis):
    reference = str(reference)
    hypothesis = str(hypothesis)

    n = len(reference)
    m = len(hypothesis)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    return dp[n][m] / max(1, n)


print(cer("我唔知道", "我不知道"))
```

What to explain:

- This is Levenshtein edit distance normalized by reference length.
- Insertions, deletions, and substitutions all count as errors.
- For Chinese/Cantonese ASR, CER is usually more reliable than WER.

### Task 4: Detect Written-Style Replacement Errors

```python
replacement_pairs = {
    "嘅": "的",
    "哋": "們",
    "唔": "不",
    "咗": "了",
    "嚟": "來",
}


def written_style_flags(reference, prediction):
    flags = []
    for spoken, written in replacement_pairs.items():
        if spoken in reference and written in prediction:
            flags.append(f"{spoken}->{written}")
    return flags


print(written_style_flags("我唔知道佢去咗邊", "我不知道他去了哪裡"))
```

What to explain:

- This does not prove the model heard wrongly.
- It detects a Cantonese transcript-style mismatch.
- This is useful for error taxonomy after baseline ASR.

### Task 5: Group CER By Metadata

```python
pred = pd.read_csv("outputs/predictions/whisper_api_dev_20.csv", encoding="utf-8")

summary = (
    pred.groupby("accents", dropna=False)
    .agg(
        rows=("path", "count"),
        mean_cer=("cer", "mean"),
        median_cer=("cer", "median"),
    )
    .sort_values("mean_cer", ascending=False)
)

summary.head(10)
```

What to explain:

- Grouped error analysis helps identify accent or metadata weakness.
- Small groups should be interpreted carefully.
- This is the first step before dialect adaptation.

## PyTorch Questions

### 1. How do you check if PyTorch can use GPU?

```python
import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
```

If `torch.cuda.is_available()` is `False`, possible reasons:

- CPU-only PyTorch was installed.
- Jupyter is using the wrong kernel.
- NVIDIA driver or CUDA-compatible wheel is missing.
- The environment used by Jupyter is not the project `.venv`.

### 2. How would you move a model and tensor to GPU?

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

model = model.to(device)
input_values = input_values.to(device)
labels = labels.to(device)
```

What to explain:

- Model and tensors must be on the same device.
- GPU helps training and inference speed, but data loading can still be a
  bottleneck.

### 3. How would you avoid GPU out-of-memory errors?

**Answer:**

I would reduce batch size first. Then I would use gradient accumulation, mixed
precision, shorter audio clips, dynamic padding, or freeze part of the encoder.
For large speech models, adapter or LoRA fine-tuning can also reduce memory
cost.

## Jupyter And Notebook Workflow Questions

### 1. Why use notebooks for this project?

**Answer:**

Notebooks are suitable for speech research because I can inspect metadata,
listen to samples, visualize distributions, run small baselines, and perform
error analysis step by step. For repeated logic such as metrics and data
loading, I keep small helper functions in `src/` so notebooks stay readable.

### 2. What should stay in `src/` instead of notebooks?

**Answer:**

Small reusable utilities should stay in `src/`, for example:

- Loading Common Voice TSV files.
- Transcript normalization.
- CER/WER calculation.
- Audio duration and loading helpers.
- API wrapper functions.

Training experiments, plots, analysis tables, and discussion can stay in
notebooks.

### 3. How would you make notebooks reproducible?

**Answer:**

I would use:

- A virtual environment.
- A named Jupyter kernel.
- Fixed dataset paths through `.env`.
- Clear notebook execution order.
- Saved prediction and analysis CSVs.
- Versioned helper code.
- Explicit random seeds for training experiments.

## Short Answers To Memorize

**Low-resource ASR:**

> Use pretrained SSL or large speech models, fine-tune on Cantonese data, apply
> augmentation, and evaluate with CER plus targeted error analysis.

**Tone modeling:**

> Add Jyutping/tone labels and pitch features as auxiliary supervision, then
> evaluate tone-related errors separately.

**Dialect adaptation:**

> Use lightweight adapters, LoRA, sparse accent embeddings, and teacher
> soft-label transfer instead of full model retraining.

**Context-aware ASR:**

> Combine acoustic features, Cantonese linguistic markers, metadata, domain
> prompts, and language model rescoring.

**Multimodal fusion:**

> Encode speech emotion and text semantics separately, then align them with
> attention or adversarial training.

**Weak supervision:**

> Use pseudo-labels carefully, clean by confidence and model agreement, and
> down-weight uncertain samples.

## One-Hour Code Test Strategy

Use your time like this:

1. First 5 minutes: read the task carefully and identify inputs, outputs, and
   metric.
2. Next 15 minutes: load data and inspect columns.
3. Next 15 minutes: implement the core function, such as CER, filtering, or
   grouping.
4. Next 10 minutes: test with a tiny example.
5. Next 10 minutes: clean variable names and add short comments only where
   useful.
6. Final 5 minutes: explain limitations and next improvements.

If you get stuck, produce a simple correct baseline first. A working simple
solution is better than an unfinished complex one.

## Questions You Can Ask The Interviewer

Good clarifying questions:

- Should the transcript preserve spoken Cantonese style, or normalize to
  standard written Chinese?
- Should CER be computed before or after punctuation normalization?
- Is the test focused on model training, data analysis, or system design?
- Are we allowed to use pretrained models or APIs?
- Should the solution optimize runtime, readability, or research flexibility?

These questions show that you understand ASR evaluation depends on transcript
policy and task definition.

## Final Project Pitch

Use this if they ask you to summarize your approach:

> My approach is baseline-first and error-analysis-driven. I start with a
> pretrained ASR model on Common Voice Yue, measure CER/WER, then inspect
> Cantonese-specific errors such as colloquial-to-written replacements and
> possible tone issues. Based on the error profile, I would add tone-aware
> auxiliary modeling, dialect adapters, soft-label transfer, and eventually
> multimodal speech-text alignment. This keeps the research practical because
> every modeling change is tied to a measurable failure case.
