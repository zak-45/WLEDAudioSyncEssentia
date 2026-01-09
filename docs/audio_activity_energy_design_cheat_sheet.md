# Audio Activity Energy – Design Cheat Sheet

This document describes the **activity_energy** signal used in the analysis core: what it measures, why it works, and how to tune it safely.

---

## What activity_energy Represents

**activity_energy ∈ [0, 1]** is a perceptual measure of *musical physical energy*:

> **Energy = Loudness × Motion**

It answers:
- *Is sound present?* (RMS)
- *Is it doing something?* (envelope motion)

This avoids classic pitfalls:
- Loud but static audio ≠ energetic
- Quiet but rhythmic audio ≠ silence

---

## Signal Flow Overview

```
FAST BUFFER (≈ 0.5–1.0s)
   ↓
Short-term RMS envelope (20 ms / 10 ms hop)
   ↓
Envelope motion = mean(|Δ envelope|)
   ↓
Normalize loudness + motion
   ↓
activity = rms_norm × motion_norm
   ↓
Temporal smoothing (EMA)
```

---

## Core Components

### 1. RMS Envelope (Loudness)
- Window: **20 ms**
- Hop: **10 ms**
- Robust against waveform polarity

Used only to ensure:
- silence stays at 0
- quiet passages don’t dominate visuals

---

### 2. Envelope Motion (Activity)

```
env_motion = mean(abs(diff(envelope)))
```

Captures:
- transients
- rhythm density
- articulation

Not affected by absolute volume.

---

### 3. Adaptive Motion Floor (Noise-Aware)

Purpose:
- Remove background hiss, mic noise, room tone

Method:
- Track motion history
- Use lower percentile as noise estimate

```
floor = percentile(motion_history, 20) × 1.8
floor clamped to [0.0005, 0.0030]
```

Key property:
> Floor adapts to environment, not music style

---

### 4. Motion Reference (Semantic Anchor)

```
MOTION_REF ≈ 0.015
```

This defines:
- what "full activity" means
- consistent behavior across systems

Rule:
- Floor adapts
- Reference stays stable

---

### 5. Final Activity Formula

```
rms_norm    = normalize(rms)
motion_norm = normalize(env_motion - floor)

activity_raw = rms_norm × motion_norm
```

Why multiplication:
- silence → 0
- loud but static → low
- rhythmic but quiet → moderate
- loud + rhythmic → high

---

### 6. Temporal Smoothing

EMA (no accumulation):

```
activity = α × current + (1 - α) × previous
```

Typical α:
- **0.15–0.25** visuals
- **0.3–0.5** UI meters

---

## Tuning Guide

### MOTION_REF
| Use case | Value |
|-------|------|
| Ambient / cinematic | 0.010–0.012 |
| General music | **0.015** |
| EDM / techno | 0.018–0.022 |

---

### Floor Strategy

✔ Adaptive floor (recommended)
- survives mic swaps
- survives room noise
- survives silence gaps

✘ Fixed floor
- brittle
- genre-dependent

---

## Common Failure Modes (and Why You Avoided Them)

❌ RMS-only energy
- confuses loudness with activity

❌ Accumulating energy
- leads to permanent saturation

❌ Fast vs slow envelope contrast
- genre biased
- inverted behavior on intros

✔ Your approach
- perceptual
- stable
- genre-agnostic

---

## Mental Model (Keep This)

> **Activity is not how loud music is**
> **Activity is how much the sound is changing**

Loudness just gates it.

---

## When to Change This System

Only if you want:
- beat-synchronous energy
- danceability replacement
- per-band activity (bass vs highs)

Otherwise: **don’t touch it** 🙂

---

## Status

✅ Production-ready
✅ Visual-safe
✅ Musically meaningful

This is a *good* signal.

