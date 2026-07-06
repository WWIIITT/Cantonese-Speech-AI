# Job Duties Code Test Questions And Answers

This document is not tied to any specific repository. It prepares you for code
test and technical interview questions based directly on the advertised duties:
Cantonese ASR, self-supervised learning, tone modeling, dialect adaptation,
context-aware recognition, multimodal alignment, weak supervision, and R&D
technical leadership.

## How To Use This Document

For a one-hour assessment, practice three types of answers:

- **Coding answer:** write a small working Python function or Pandas analysis.
- **Concept answer:** explain the idea clearly in two or three minutes.
- **System answer:** describe a practical framework with data, model,
  training, evaluation, and limitations.

The strongest answers usually follow this shape:

```text
Problem -> Method -> Why it helps -> How to evaluate -> Limitation
```

## Part 1: Cantonese ASR And Low-Resource Learning

### Question 1. What makes Cantonese ASR difficult?

**Concept answer:**

Cantonese ASR is difficult because Cantonese has limited labeled speech data,
lexical tones, strong differences between spoken Cantonese and written Chinese,
speaker and accent variation, and noisy real-world recording conditions. A
general Chinese ASR model may output standard written Chinese instead of
spoken Cantonese. For example, it may output `不` instead of Cantonese `唔`.

**Stronger answer:**

I would separate the problem into acoustic difficulty and linguistic difficulty.
The acoustic side includes tone, pitch contour, speaker variation, recording
quality, and background noise. The linguistic side includes colloquial
Cantonese words, code-switching, no explicit word boundaries in Chinese text,
and mismatch between spoken Cantonese transcripts and written Chinese output.
Because of these issues, I would evaluate not only average CER, but also
tone-sensitive errors, colloquial marker errors, and accent-specific errors.

### Question 2. How would you develop Cantonese ASR for a low-resource setting?

**System answer:**

I would not train from scratch. I would start from a pretrained speech model,
such as wav2vec 2.0, HuBERT, WavLM, or Whisper. These models already learned
general acoustic representations from large-scale audio. Then I would fine-tune
or adapt the model on available Cantonese data.

Pipeline:

1. Collect Cantonese audio and transcripts.
2. Normalize transcripts conservatively.
3. Split train/dev/test by speaker if possible.
4. Build a baseline using Whisper or a CTC model.
5. Evaluate with CER as the primary metric.
6. Analyze errors by tone, duration, noise, speaker, and accent.
7. Add augmentation and tone-aware modeling.
8. Use lightweight adaptation for accent or domain changes.

**Why this works:**

Self-supervised or pretrained models reduce the amount of labeled Cantonese
data needed. Error analysis makes sure every model change is connected to a
real failure mode.

### Question 3. What is self-supervised learning in speech?

**Concept answer:**

Self-supervised learning learns from unlabeled speech. The model creates a
training signal from the audio itself, for example by masking part of the audio
representation and predicting the missing part. This lets the model learn
useful acoustic representations without needing transcripts for every audio
file.

Examples:

- **wav2vec 2.0:** masks latent speech representations and predicts quantized
  speech units.
- **HuBERT:** predicts cluster labels for masked speech frames.
- **WavLM:** improves robustness for speech tasks such as ASR and speaker
  processing.

**Interview phrasing:**

Self-supervised learning is valuable for low-resource Cantonese because
unlabeled speech is easier to obtain than accurately transcribed Cantonese
speech. I would use SSL pretraining or pretrained SSL models, then fine-tune on
the smaller labeled Cantonese dataset.

### Question 4. Code a simple Character Error Rate function.

**Coding answer:**

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

**Concept explanation:**

CER is edit distance divided by the number of reference characters. It counts
substitutions, deletions, and insertions. CER is especially useful for Chinese
and Cantonese because word segmentation is ambiguous.

### Question 5. Why use CER instead of WER for Cantonese?

**Answer:**

WER works well for English because words are separated by spaces. Cantonese and
Chinese text do not naturally include spaces, so WER depends on the word
segmentation tool. Different segmentation rules can produce different WER
values. CER avoids this problem by comparing character by character, so it is
more stable for Cantonese ASR.

I may still report WER as a secondary metric if the segmentation method is
fixed and documented.

## Part 2: Tone Modeling

### Question 6. Why is tone modeling important for Cantonese ASR?

**Concept answer:**

Cantonese is a tonal language. The pitch contour of a syllable can change the
meaning of a word. If the model ignores tone, it may confuse words with similar
consonants and vowels but different tones. This is especially important in
noisy or low-resource conditions where acoustic evidence is weak.

**Practical answer:**

I would add tone modeling as an auxiliary objective. The ASR model predicts the
main transcript, while a tone branch predicts tone labels. I would also extract
pitch or F0 features to help represent prosody.

### Question 7. How would you create tone labels?

**Answer:**

I would convert Cantonese characters into Jyutping with tone numbers, such as:

```text
你 -> nei5
知 -> zi1
道 -> dou6
```

The tone number can be extracted as the tone label. For example, `nei5` has
tone `5`.

However, automatic conversion is not perfect because Cantonese has polyphonic
characters and context-dependent pronunciation. I would treat these labels as
useful weak labels, not perfect ground truth, unless a verified pronunciation
lexicon is available.

### Question 8. Code tone-number extraction from Jyutping strings.

**Coding answer:**

```python
import re


def extract_tones(jyutping_text):
    return re.findall(r"[1-6]", str(jyutping_text))


print(extract_tones("nei5 hou2 maa3"))
```

**Expected output:**

```text
['5', '2', '3']
```

**What to explain:**

This is a simple helper for tone analysis. In a full system, I would align
tones with characters or syllables more carefully.

### Question 9. How would you combine ASR loss and tone loss?

**Answer:**

I would use multi-task learning:

```text
total_loss = asr_loss + lambda_tone * tone_loss
```

The ASR loss can be CTC loss or sequence-to-sequence loss. The tone loss can be
cross-entropy over tone classes. The value of `lambda_tone` controls how much
the tone task influences training.

**Limitation:**

If tone labels are noisy, a large tone-loss weight can hurt ASR. I would tune
the weight on the development set.

### Question 10. What are prosody features?

**Answer:**

Prosody features describe how speech is spoken beyond phonetic content. Common
features include:

- Pitch or F0 contour.
- Energy or loudness.
- Speaking rate.
- Pause duration.
- Rhythm.

For Cantonese, pitch contour is especially relevant because it relates to tone.
In noisy speech, prosody can also provide clues about emotion, sentence
structure, and speaker intent.

## Part 3: Dialect Adaptation

### Question 11. What is dialect-adaptive ASR?

**Concept answer:**

Dialect-adaptive ASR means the model can adapt to different accents or dialect
variations without retraining the whole model from scratch. For Cantonese, this
may include Hong Kong Cantonese, Guangdong Cantonese, overseas speakers, or
different speaker groups.

**System answer:**

I would start by measuring CER by accent group. If some groups perform worse, I
would use lightweight adaptation methods such as adapter layers, LoRA, accent
embeddings, or sparse routing.

### Question 12. What is sparse encoding for accent adaptation?

**Answer:**

Sparse encoding means the model uses a compact representation or activates only
a subset of parameters for a given accent. Instead of updating the whole model,
we learn small accent-specific components.

Example:

```text
Base ASR model: shared by all speakers
Adapter A: Hong Kong accent
Adapter B: Guangdong accent
Adapter C: overseas accent
```

This reduces training cost and helps avoid overfitting when accent-specific
data is small.

### Question 13. What is soft-label transfer?

**Answer:**

Soft-label transfer is a teacher-student method. A strong teacher model
produces probability distributions, and a smaller or adapted student model
learns from those distributions.

Hard label:

```text
Correct token: 唔
```

Soft label:

```text
唔: 0.72
不: 0.15
冇: 0.06
other: 0.07
```

Soft labels contain uncertainty. They can teach the student that some mistakes
are more plausible than others.

### Question 14. How would you code a simple confidence filter for pseudo labels?

**Coding answer:**

```python
import pandas as pd

df = pd.DataFrame(
    {
        "audio_id": ["a1", "a2", "a3"],
        "pseudo_text": ["你好", "我唔知", "今日天氣好"],
        "confidence": [0.95, 0.62, 0.88],
    }
)

clean = df[df["confidence"] >= 0.8].copy()
print(clean)
```

**What to explain:**

Confidence filtering is a simple pseudo-label cleaning method. I would tune the
threshold on validation data. A high threshold gives cleaner labels but fewer
training samples.

### Question 15. How would you evaluate dialect adaptation?

**Answer:**

I would compare baseline and adapted models using:

- Overall CER.
- CER by accent group.
- CER by speaker group.
- CER by duration bucket.
- Error categories such as tone errors or written-style replacements.

The adapted model should improve weak accent groups without significantly
hurting strong groups.

## Part 4: Context-Aware ASR

### Question 16. What is a context-aware ASR framework?

**Answer:**

A context-aware ASR framework uses more than raw audio. It can include
linguistic context, speaker metadata, domain vocabulary, tone information,
prosody, and noise condition.

For example:

- Use a language model to rescore ASR hypotheses.
- Add domain-specific vocabulary.
- Prefer Cantonese colloquial words when the transcript style requires them.
- Use pitch/prosody features for tone-sensitive recognition.
- Use accent metadata for adaptation.

### Question 17. How would you improve ASR in noisy Cantonese speech?

**Answer:**

I would use data augmentation and robust evaluation.

Training methods:

- Add background noise.
- Apply speed perturbation.
- Apply volume perturbation.
- Use SpecAugment.
- Mix clean and noisy samples.

Evaluation:

- Report CER on clean and noisy subsets.
- Group CER by signal-to-noise ratio if available.
- Inspect short and long clips separately.

### Question 18. Code a simple duration bucket analysis.

**Coding answer:**

```python
import pandas as pd

df = pd.DataFrame(
    {
        "duration_sec": [1.2, 4.5, 8.0, 15.2, 25.0],
        "cer": [0.4, 0.2, 0.1, 0.3, 0.5],
    }
)

df["duration_bucket"] = pd.cut(
    df["duration_sec"],
    bins=[0, 3, 6, 10, 20, float("inf")],
    labels=["0-3s", "3-6s", "6-10s", "10-20s", "20s+"],
)

summary = df.groupby("duration_bucket", observed=False)["cer"].mean()
print(summary)
```

**What to explain:**

Duration can affect ASR performance. This analysis helps identify whether
errors are concentrated in very short or very long clips.

## Part 5: Multimodal Alignment

### Question 19. What is multimodal speech-text alignment?

**Concept answer:**

Multimodal alignment means mapping information from different modalities, such
as speech audio and text, into a shared or compatible representation space. In
this job duty, the model should fuse speech emotion features with text
semantics.

Speech may contain emotion through pitch, energy, speed, and pauses. Text
contains semantic meaning. Combining both can improve emotion understanding,
intent recognition, or context-aware ASR.

### Question 20. How would you build attention-based emotional embedding?

**System answer:**

I would use two encoders:

- Speech encoder: extracts acoustic/emotion features.
- Text encoder: extracts semantic embeddings.

Then I would use attention:

```text
speech_embedding = SpeechEncoder(audio)
text_embedding = TextEncoder(text)
fused_embedding = CrossAttention(query=text_embedding, key=speech_embedding, value=speech_embedding)
```

The attention module learns which parts of the speech emotion representation
are relevant to the text meaning. The fused embedding can be used for emotion
classification or multimodal understanding.

### Question 21. Code a simple PyTorch attention fusion module.

**Coding answer:**

```python
import torch
import torch.nn as nn


class AttentionFusion(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, text_emb, speech_emb):
        attended, _ = self.attn(
            query=text_emb,
            key=speech_emb,
            value=speech_emb,
        )
        return self.norm(text_emb + attended)


batch_size = 2
text_len = 5
speech_len = 12
dim = 64

text_emb = torch.randn(batch_size, text_len, dim)
speech_emb = torch.randn(batch_size, speech_len, dim)

fusion = AttentionFusion(dim)
out = fusion(text_emb, speech_emb)
print(out.shape)
```

**Expected output:**

```text
torch.Size([2, 5, 64])
```

**What to explain:**

The text embedding queries the speech embedding. This lets the model select
speech emotion information that is relevant to each text token or sentence
representation.

## Part 6: Weak Supervision And Adversarial Alignment

### Question 22. What is weakly supervised multimodal fusion?

**Answer:**

Weakly supervised multimodal fusion means training a multimodal model when the
labels are noisy, incomplete, or automatically generated. For example, emotion
labels may come from a pretrained classifier instead of human annotation.

The challenge is that noisy labels can mislead the model, so pseudo-label
cleaning and robust training are important.

### Question 23. How would you clean pseudo labels?

**Answer:**

I would use several cleaning strategies:

- Keep only high-confidence pseudo labels.
- Remove samples where audio and text models strongly disagree.
- Down-weight uncertain samples instead of deleting all of them.
- Remove outliers with abnormal duration or low audio quality.
- Manually inspect a small validation subset.

The trade-off is precision versus coverage. Stricter filtering gives cleaner
data but fewer samples.

### Question 24. What is adversarial alignment?

**Concept answer:**

Adversarial alignment uses a discriminator to encourage embeddings from
different modalities or domains to become similar. The discriminator tries to
predict whether an embedding comes from speech or text. The encoder tries to
make this prediction difficult.

Training idea:

```text
Discriminator: distinguish speech embedding vs text embedding
Encoder: produce embeddings that fool the discriminator
```

This can help align speech emotion features and text semantic features in a
shared representation space.

### Question 25. Code a simple domain discriminator.

**Coding answer:**

```python
import torch
import torch.nn as nn


class DomainDiscriminator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 2),
        )

    def forward(self, x):
        return self.net(x)


dim = 128
embeddings = torch.randn(8, dim)
domain_labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

disc = DomainDiscriminator(dim)
logits = disc(embeddings)
loss = nn.CrossEntropyLoss()(logits, domain_labels)

print(logits.shape)
print(loss.item())
```

**What to explain:**

This discriminator predicts whether an embedding belongs to one domain or
another. In adversarial training, the encoder is trained to make this task
difficult, encouraging domain-invariant representations.

## Part 7: PyTorch And Training Questions

### Question 26. How do you check GPU availability?

**Coding answer:**

```python
import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

**What to explain:**

If `torch.cuda.is_available()` is false, I would check whether I installed a
CPU-only PyTorch build, whether Jupyter is using the correct kernel, and whether
the NVIDIA driver is available.

### Question 27. Show a minimal PyTorch training loop.

**Coding answer:**

```python
import torch
import torch.nn as nn

model = nn.Linear(10, 2)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

x = torch.randn(32, 10)
y = torch.randint(0, 2, (32,))

model.train()
optimizer.zero_grad()
logits = model(x)
loss = loss_fn(logits, y)
loss.backward()
optimizer.step()

print(loss.item())
```

**What to explain:**

The key steps are forward pass, loss calculation, backpropagation, and optimizer
update. For a real ASR model, the input would be audio features and the loss
could be CTC or sequence-to-sequence loss.

### Question 28. How would you avoid overfitting in low-resource speech tasks?

**Answer:**

I would use:

- Pretrained models.
- Data augmentation.
- Early stopping.
- Dropout.
- Freezing part of the encoder.
- Adapter or LoRA tuning.
- Validation by speaker or accent group.
- Careful error analysis instead of only training longer.

Low-resource ASR can overfit quickly, so validation design matters.

## Part 8: R&D Leadership

### Question 29. How would you guide junior R&D team members?

**Answer:**

I would give the team a reproducible workflow:

- Start with a simple baseline.
- Define metrics before improving the model.
- Save predictions for error analysis.
- Review data quality and transcript policy.
- Use small experiments before full training.
- Document assumptions and limitations.
- Use code review to check correctness, not only style.

For mentoring, I would explain both the modeling concept and the engineering
reason behind each step.

### Question 30. How would you review a teammate's ASR experiment?

**Answer:**

I would check:

- Is the dataset split correct?
- Is there speaker leakage between train and test?
- Are transcripts normalized consistently?
- Are CER/WER computed correctly?
- Are results compared against a baseline?
- Are failure examples inspected?
- Are hyperparameters and model versions recorded?
- Can another team member reproduce the experiment?

Good R&D review should protect both scientific validity and engineering
reproducibility.

## Final Interview Pitch

If they ask you to summarize your suitability for the duties, say:

> I would approach the role with a baseline-first and error-analysis-driven
> workflow. For low-resource Cantonese ASR, I would start from pretrained
> self-supervised speech models, evaluate with CER, and analyze Cantonese
> specific failures such as tone errors and spoken-to-written replacements.
> Then I would add tone-aware auxiliary modeling, prosody features,
> dialect-adaptive adapters, sparse accent representations, and soft-label
> transfer. For multimodal work, I would align speech emotion features and text
> semantic embeddings with attention and adversarial training, while using
> pseudo-label cleaning to control weak-label noise. I would also make the work
> reproducible for the R&D team through notebooks, shared utilities, experiment
> tracking, and code review.

## One-Hour Code Test Strategy

Use this strategy during the assessment:

1. Read the task and identify input, output, and metric.
2. Load and inspect the data first.
3. Write a simple correct baseline.
4. Test with a tiny example.
5. Add grouping or analysis only after the core function works.
6. Explain limitations clearly.

If the test is open-ended, choose a small complete solution over a large
unfinished one.
