# Code Test Questions And Answers

This document focuses on practical code-test questions for a Cantonese speech
AI engineer interview. It is written for a one-hour laptop assessment where the
task may involve Pandas, Jupyter, PyTorch, speech metadata, ASR metrics, or
high-level model design.

## 1. Load A Common Voice TSV File

**Question:**

You are given a Common Voice Cantonese `dev.tsv`. Load it with Pandas and show
the first few audio paths and sentences.

**Answer:**

```python
from pathlib import Path
import pandas as pd

root = Path("Mozilla-HK-Speech-datasets/cv-corpus-26.0-2026-06-12/yue")
dev = pd.read_csv(root / "dev.tsv", sep="\t", encoding="utf-8")

print(dev.shape)
dev[["path", "sentence", "accents", "age", "gender"]].head()
```

**What to explain:**

Common Voice metadata files are tab-separated, so `sep="\t"` is required.
Because the transcripts contain Cantonese characters, I explicitly use
`encoding="utf-8"`. The `path` column points to files under the `clips/`
folder, while metadata such as `accents`, `age`, and `gender` can be used later
for dialect and subgroup error analysis.

## 2. Create Full Audio Paths

**Question:**

Add a full `audio_path` column for each row in the metadata.

**Answer:**

```python
from pathlib import Path

clips_dir = root / "clips"
dev["audio_path"] = dev["path"].apply(lambda name: str(clips_dir / name))

dev[["path", "audio_path", "sentence"]].head()
```

**What to explain:**

During data discovery, I only construct file paths instead of loading all audio
waveforms. Loading every MP3 immediately would be slow and memory-heavy. I
would load audio lazily inside training, inference, or audio inspection code.

## 3. Check Missing Audio Files

**Question:**

Check how many metadata rows point to missing audio files.

**Answer:**

```python
from pathlib import Path

dev["audio_exists"] = dev["audio_path"].apply(lambda p: Path(p).exists())
missing_count = (~dev["audio_exists"]).sum()

print("missing audio:", missing_count)
```

**What to explain:**

This is a basic data-quality check. If audio files are missing, I would not
crash the whole pipeline. I would report the missing count, remove those rows
for training, and keep the report for reproducibility.

## 4. Join Clip Duration Metadata

**Question:**

Join `clip_durations.tsv` into the speech metadata and create `duration_sec`.

**Answer:**

```python
durations = pd.read_csv(root / "clip_durations.tsv", sep="\t", encoding="utf-8")

duration_col = "duration[ms]" if "duration[ms]" in durations.columns else "duration_ms"
durations = durations.rename(columns={"clip": "path", duration_col: "duration_ms"})

df = dev.merge(durations[["path", "duration_ms"]], on="path", how="left")
df["duration_sec"] = df["duration_ms"] / 1000

df[["path", "sentence", "duration_sec"]].head()
```

**What to explain:**

Duration is useful for filtering and error analysis. Very short clips may be
hard to recognize, while very long clips may cause memory issues during model
training. I would inspect duration distribution before training.

## 5. Basic Transcript Normalization

**Question:**

Write a simple function to normalize Cantonese transcripts.

**Answer:**

```python
import re


def normalize_transcript(text):
    text = str(text).strip().lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text)
    return text


df["normalized_sentence"] = df["sentence"].apply(normalize_transcript)
df[["sentence", "normalized_sentence"]].head()
```

**What to explain:**

For Cantonese, normalization should be conservative. I remove spaces and
punctuation, but I keep Chinese characters, English letters, and numbers. I
would not blindly convert spoken Cantonese words into written Chinese, because
`唔` and `不` are different transcript styles for this task.

## 6. Implement Character Error Rate

**Question:**

Implement CER without using an external library.

**Answer:**

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

**What to explain:**

CER is edit distance divided by the number of reference characters. It counts
substitutions, deletions, and insertions. For Cantonese and Chinese ASR, CER is
often more reliable than WER because Chinese text has no explicit word
boundaries.

## 7. Explain CER With A Cantonese Example

**Question:**

Explain why `我唔知道` vs `我不知道` has an error.

**Answer:**

```text
Reference:  我唔知道
Prediction: 我不知道
```

The model changed `唔` into `不`. Semantically, both may mean "not", but the
reference is spoken Cantonese and the prediction is written Chinese. For a
Cantonese ASR transcript task, this is an error.

If the reference has four characters and one character differs, the CER is:

```text
1 / 4 = 0.25
```

## 8. Compute CER For A Prediction CSV

**Question:**

Given a CSV with `sentence` and `prediction`, compute CER for every row.

**Answer:**

```python
pred = pd.read_csv("outputs/predictions/whisper_api_dev_20.csv", encoding="utf-8")

pred["normalized_sentence"] = pred["sentence"].apply(normalize_transcript)
pred["normalized_prediction"] = pred["prediction"].apply(normalize_transcript)

pred["cer"] = pred.apply(
    lambda row: cer(row["normalized_sentence"], row["normalized_prediction"]),
    axis=1,
)

print(pred["cer"].mean())
pred[["sentence", "prediction", "cer"]].sort_values("cer", ascending=False).head()
```

**What to explain:**

I normalize both reference and prediction before computing CER, otherwise
punctuation and spacing may dominate the metric. After computing CER, I inspect
the worst examples because average metrics alone do not explain model failure.

## 9. Detect Cantonese-To-Written-Chinese Replacements

**Question:**

Detect cases where the reference contains spoken Cantonese but the prediction
uses written Chinese.

**Answer:**

```python
replacement_pairs = {
    "嘅": "的",
    "哋": "們",
    "唔": "不",
    "咗": "了",
    "嚟": "來",
    "畀": "給",
}


def written_replacement_flags(reference, prediction):
    flags = []
    for spoken, written in replacement_pairs.items():
        if spoken in str(reference) and written in str(prediction):
            flags.append(f"{spoken}->{written}")
    return flags


pred["written_replacement_flags"] = pred.apply(
    lambda row: written_replacement_flags(row["sentence"], row["prediction"]),
    axis=1,
)

pred[["sentence", "prediction", "written_replacement_flags"]].head()
```

**What to explain:**

This creates an interpretable Cantonese-specific error feature. It does not
replace CER, but it helps explain why a model may be semantically close while
still failing the target transcript style.

## 10. Group Error Rate By Accent

**Question:**

Find which accent group has the highest average CER.

**Answer:**

```python
accent_summary = (
    pred.groupby("accents", dropna=False)
    .agg(
        rows=("path", "count"),
        mean_cer=("cer", "mean"),
        median_cer=("cer", "median"),
    )
    .sort_values("mean_cer", ascending=False)
)

accent_summary.head(10)
```

**What to explain:**

This is the first step toward dialect adaptation. If a certain accent group has
higher CER, I would inspect whether the group is large enough, then consider
adapter tuning, accent embeddings, or data augmentation. I would be careful
with very small groups because their mean CER may be unstable.

## 11. Group Error Rate By Duration

**Question:**

Analyze ASR performance by audio duration bucket.

**Answer:**

```python
pred["duration_bucket"] = pd.cut(
    pred["duration_sec"],
    bins=[0, 3, 6, 10, 20, float("inf")],
    labels=["0-3s", "3-6s", "6-10s", "10-20s", "20s+"],
)

duration_summary = (
    pred.groupby("duration_bucket", observed=False)
    .agg(rows=("path", "count"), mean_cer=("cer", "mean"))
)

duration_summary
```

**What to explain:**

Duration affects ASR quality. Short utterances may lack context, while long
utterances may increase decoding difficulty. Duration-based analysis helps
decide filtering and batching strategies.

## 12. Check PyTorch GPU Availability

**Question:**

Write code to check whether PyTorch can use the GPU.

**Answer:**

```python
import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

**What to explain:**

If `torch.cuda.is_available()` is `False`, the most common causes are:

- CPU-only PyTorch is installed.
- Jupyter is using the wrong kernel.
- NVIDIA driver is missing or incompatible.
- The project virtual environment is not selected.

## 13. Move A Model And Tensors To GPU

**Question:**

Show the standard PyTorch device pattern.

**Answer:**

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
input_values = input_values.to(device)
labels = labels.to(device)

outputs = model(input_values, labels=labels)
loss = outputs.loss
```

**What to explain:**

The model and tensors must be on the same device. A common error is moving the
model to GPU but leaving input tensors on CPU.

## 14. Avoid GPU Out-Of-Memory

**Question:**

What would you do if speech model training runs out of GPU memory?

**Answer:**

I would first reduce batch size. If that is not enough, I would use gradient
accumulation, mixed precision, shorter maximum audio length, dynamic padding,
freezing part of the encoder, or adapter/LoRA fine-tuning instead of full
fine-tuning.

For speech models, audio length strongly affects memory usage, so filtering or
bucketed batching by duration can help.

## 15. Explain CTC Loss

**Question:**

Why is CTC loss useful for ASR?

**Answer:**

CTC loss is useful because we usually have audio and transcript pairs, but not
frame-level character alignment. The audio may have hundreds or thousands of
frames, while the transcript has only a few characters.

CTC allows the model to output blank tokens and repeated tokens, then collapse
them into a final transcript.

```text
Frame output: 我 _ 唔 唔 _ 知 知 道
Collapsed:    我唔知道
```

This avoids needing manually aligned labels.

## 16. Design A Low-Resource Cantonese ASR Pipeline

**Question:**

Describe your full ASR pipeline for Cantonese.

**Answer:**

I would use a baseline-first workflow:

1. Load Common Voice Yue metadata and audio paths.
2. Normalize transcripts conservatively.
3. Build a Whisper or wav2vec/HuBERT baseline.
4. Evaluate with CER and WER.
5. Analyze Cantonese-specific errors, especially colloquial-to-written
   replacements.
6. Add tone modeling using Jyutping labels and pitch/F0 features.
7. Add dialect adaptation using adapters, LoRA, sparse accent embeddings, or
   soft-label transfer.
8. Re-evaluate by accent, duration, age, gender, and error taxonomy.

The key is to make every modeling improvement respond to a measured error
pattern.

## 17. Explain Self-Supervised Speech Learning

**Question:**

What is self-supervised learning, and why is it useful for Cantonese ASR?

**Answer:**

Self-supervised learning uses unlabeled audio to learn speech representations.
For example, wav2vec 2.0 masks parts of the speech representation and learns to
predict the missing units. HuBERT predicts cluster labels for masked speech
frames.

This is useful for Cantonese because labeled Cantonese speech is limited.
Instead of training from scratch, I can fine-tune a pretrained SSL model on a
smaller Cantonese dataset.

## 18. Add Tone Modeling

**Question:**

How would you add Cantonese tone modeling to ASR?

**Answer:**

I would add tone modeling as an auxiliary objective:

```text
total_loss = asr_loss + lambda_tone * tone_loss
```

The ASR branch predicts characters. The tone branch predicts tone labels from
Jyutping or pitch-related features. I would also extract F0 contours from audio
and compare whether tone-aware training improves tone-sensitive errors.

I would treat automatic Jyutping labels carefully because polyphonic characters
and context can make tone conversion imperfect.

## 19. Dialect Adaptation With Soft Labels

**Question:**

How would you adapt an ASR model to accents quickly?

**Answer:**

I would avoid full retraining. I would use lightweight adapter layers or LoRA
modules and train them on accent-specific data. If data is limited, I would use
a stronger teacher model to generate soft labels.

Soft labels are probability distributions, not only hard labels. They preserve
uncertainty, for example:

```text
唔: 0.72
不: 0.16
冇: 0.05
other: 0.07
```

This helps the student model learn from a teacher without overfitting a small
accent dataset.

## 20. Multimodal Fusion Design

**Question:**

How would you fuse speech emotion features with text semantics?

**Answer:**

I would build two encoders:

- A speech encoder for acoustic or emotion features such as pitch, energy,
  speaking rate, or pretrained emotion embeddings.
- A text encoder for semantic embeddings from BERT, RoBERTa, or an LLM
  embedding model.

Then I would use attention-based fusion. For example, text embeddings can
attend to speech emotion embeddings, or both modalities can be fused with
cross-attention. The fused representation can be used for emotion
classification, intent recognition, or context-aware ASR correction.

## 21. Weak Supervision And Pseudo-Label Cleaning

**Question:**

How would you train when emotion labels or ASR labels are noisy?

**Answer:**

I would use weak supervision carefully. Pseudo-labels can come from pretrained
ASR or emotion models, but I would clean them before training.

Cleaning methods:

- Keep only high-confidence predictions.
- Remove samples where multiple models disagree.
- Down-weight uncertain samples.
- Remove samples with very high ASR error.
- Manually inspect a small validation subset.

This reduces the risk of training the model on wrong labels.

## 22. Adversarial Alignment

**Question:**

What is adversarial alignment in multimodal learning?

**Answer:**

Adversarial alignment tries to make embeddings from different modalities or
domains more similar. A discriminator tries to tell whether an embedding comes
from speech or text. The encoder tries to fool the discriminator.

This can help speech emotion embeddings and text semantic embeddings align in a
shared space, especially when labels are weak or noisy.

## 23. Debug A Bad ASR Baseline

**Question:**

Your ASR baseline has high CER. What do you check first?

**Answer:**

I would check:

1. Are audio paths correct?
2. Are transcripts loaded with correct UTF-8 encoding?
3. Is the sample rate correct?
4. Is transcript normalization too aggressive?
5. Are reference and prediction columns aligned?
6. Are there many written-style replacements?
7. Are errors concentrated in certain accents or duration buckets?
8. Is the model language setting suitable for Cantonese/Yue?

I would inspect several worst-CER examples before changing the model.

## 24. Explain Your Notebook Structure

**Question:**

Why did you split your project into several notebooks?

**Answer:**

The notebooks represent a research workflow:

- `01_data_discovery.ipynb`: inspect dataset quality and metadata.
- `02_asr_baseline.ipynb`: generate baseline ASR predictions.
- `03_tone_modeling.ipynb`: analyze Cantonese colloquial and tone-related
  errors.
- `04_dialect_adaptation.ipynb`: identify groups that need adaptation.
- `05_multimodal_fusion.ipynb`: design speech-text fusion features.
- `06_error_analysis.ipynb`: summarize error patterns.

This keeps experiments readable while reusable code such as metrics and data
loading stays in `src/`.

## 25. Final Two-Minute Answer

**Question:**

Summarize your approach to this job's technical duties.

**Answer:**

My approach is baseline-first and error-analysis-driven. For Cantonese ASR, I
would begin with pretrained speech models such as Whisper, wav2vec 2.0, HuBERT,
or WavLM because Cantonese labeled data is limited. I would evaluate with CER,
then analyze errors specific to Cantonese, including tone-sensitive mistakes
and spoken-to-written replacements such as `唔` to `不`.

After the baseline, I would improve the system with tone-aware auxiliary
training, pitch/prosody features, and dialect adaptation using adapters or
LoRA. For rapid accent adaptation, I would use sparse encoding and soft-label
transfer from a stronger teacher model. For multimodal work, I would fuse speech
emotion embeddings and text semantic embeddings using attention, then clean
weak pseudo-labels and use adversarial alignment to improve representation
consistency.

The important principle is that each model improvement should be connected to a
measurable error pattern, not added blindly.

## Fast Practice Checklist

Before the code test, make sure you can do these without notes:

- Read a TSV with Pandas.
- Build a full audio path from a metadata row.
- Join duration metadata.
- Normalize Cantonese transcript text.
- Implement CER using dynamic programming.
- Group mean CER by accent or duration.
- Explain why CER is preferred over WER for Cantonese.
- Check PyTorch GPU availability.
- Explain CTC loss in simple terms.
- Explain tone modeling, soft-label transfer, and weak supervision.
