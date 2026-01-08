# Energy Model Documentation

This document explains **how energy is defined, computed, and used** in the analysis pipeline.

It clarifies the distinction between **activity energy** and **emotional energy**, why both are necessary, and how their math was designed to behave correctly across genres and moods.

---

## 1. Why “Energy” Needs to Be Split

In music perception, **energy is not a single dimension**.

Humans perceive at least two different kinds of intensity:

| Type | Meaning | Example |
|----|----|----|
| **Activity Energy** | Physical motion, rhythm, movement | Rock, EDM, drums |
| **Emotional Energy** | Emotional intensity, tension, feeling | Sad ballads, cinematic builds |

If these are mixed into a single metric, the system fails on:
- Sad but intense songs (*Hurt*)
- Calm but emotional tracks
- Fast but emotionally shallow music

Therefore:

> **Activity energy and emotional energy must be computed separately.**

---

## 2. Activity Energy

### 2.1 Definition

**Activity energy** measures:

> *How much physical motion is present in the sound.*

It is **not loudness**, and it is **not genre**.

It answers:
- Is the signal moving rapidly?
- Are there transients and rhythmic change?

---

### 2.2 Components

Activity energy is computed from **short-term audio physics**:

#### 1️⃣ RMS (Gate only)

```text
Purpose: Silence detection
```

RMS is used **only** to ignore silence.

It does **not** drive energy.

---

#### 2️⃣ Temporal Motion (Primary)

Computed as the RMS of the **first derivative** of the waveform:

```python
diff = np.diff(segment)
motion = sqrt(mean(diff²))
```

This captures:
- Speed of change
- Transient activity
- Rhythmic movement

This is the **dominant signal**.

---

#### 3️⃣ Transient Density (Secondary)

Estimated via zero-crossing rate:

```python
zcr = mean(abs(diff(sign(segment)))) * 0.5
```

This captures:
- Percussiveness
- Texture density

---

### 2.3 Normalization

Both motion and ZCR are normalized using **perceptual reference ranges** derived from real music:

```python
motion_norm = clip((motion - 0.002) / (0.02 - 0.002), 0, 1)
zcr_norm    = clip((zcr    - 0.05 ) / (0.25 - 0.05 ), 0, 1)
```

These constants define **what humans perceive as low vs high activity**.

---

### 2.4 Fusion

Motion dominates, transients assist:

```python
raw_activity = 0.7 * motion_norm + 0.3 * zcr_norm
```

---

### 2.5 Temporal Smoothing (Critical)

Human perception integrates motion over time.

```python
activity_smooth = 0.85 * previous + 0.15 * raw_activity
```

This prevents:
- Flicker
- Overreaction to single hits

---

### 2.6 Output

```text
activity_energy ∈ [0, 1]
```

Interpretation:

| Value | Meaning |
|----|----|
| 0.0 | Silence / stillness |
| 0.3 | Calm movement |
| 0.6 | Active music |
| 1.0 | Max physical intensity |

---

## 3. Emotional Energy

### 3.1 Definition

**Emotional energy** measures:

> *How emotionally intense the moment feels.*

It is **not happiness**, **not sadness**, and **not loudness**.

Emotion intensity comes from:
- Strength of emotion (positive or negative)
- Physical engagement
- Confidence of interpretation

---

### 3.2 Valence Intensity (Core)

Valence indicates emotional polarity.

But **intensity** is distance from neutral, not direction.

```python
valence_intensity = abs(valence)
```

| Valence | Meaning | Intensity |
|----|----|----|
| -0.8 | Sad | High |
|  0.0 | Neutral | Low |
| +0.8 | Joyful | High |

---

### 3.3 Fusion With Activity

Emotion and motion reinforce each other, but neither dominates fully.

```python
emotion_base = 0.6 * valence_intensity + 0.4 * activity_energy
```

This ensures:
- Sad but calm music still feels intense
- Fast but empty music doesn’t dominate emotionally

---

### 3.4 Confidence Stabilization

Classifier confidence stabilizes emotion without driving it:

```python
confidence_gain = 0.85 + 0.15 * top_conf
```

This:
- Prevents jitter
- Strengthens confident moments

---

### 3.5 Final Formula

```python
emotional_energy = emotion_base * confidence_gain
emotional_energy = clip(emotional_energy, 0, 1)
```

---

## 4. Behavioral Summary

| Scenario | Activity | Valence | Emotional Energy |
|----|----|----|----|
| Ambient drone | Low | Neutral | Low |
| Sad ballad | Low | Strong | High |
| Pop song | Medium | Positive | Medium |
| Punk rock | High | Positive | Medium–High |
| Cinematic build | Rising | Strong | Rising |

---

## 5. Design Principles

- Energy is **perceptual**, not physical
- Loudness is never energy
- Sadness can be intense
- Motion ≠ emotion
- Time smoothing is mandatory

---

## 6. Usage in the System

| Parameter | Driven By |
|----|----|
| Brightness | Emotional energy |
| Saturation | Confidence |
| Motion effects | Activity energy |
| Accent flashes | Activity + beats |
| Mood hue | Valence + emotion |

---

## 7. Final Note

This energy model is:
- Genre-agnostic
- Emotionally correct
- Stable in real time
- Artistically controllable

It is designed to behave **like a human listener**, not a meter.

---

**End of document.**

