# Emotion & Genre Color System – Parameter Guide

This document explains **all emotion-, genre-, and visualization-related parameters** used in the WLEDAudioSyncEssentia project.

It is written for:
- users tuning visuals
- developers extending the system
- live / lighting use (predictability matters)

The system is intentionally **layered**. Understanding the layers is the key to using it well.

---

## 1. High‑level architecture (mental model)

The lighting system is built from **three independent layers**:

```
AUDIO FEATURES
     ↓
EMOTION SPACE (valence / arousal / intensity)
     ↓
EMOTION COLOR (semantic meaning)
     ↓
GENRE PROFILE (style & expression)
     ↓
WLED EFFECTS + MOTION
```

### Why this matters

- **Emotion decides *what* is felt**
- **Genre decides *how strongly* it is expressed**
- **Effects decide *how it moves***

These layers are deliberately *not merged*. This keeps behavior predictable and tunable.

---

## 2. Emotion space (Essentia AUX → emotion)

Essentia AUX classifiers provide values in **[0..1]**:

| Feature | Meaning |
|------|-------|
| `happy` | positive affect |
| `sad` | negative affect |
| `relaxed` | low energy |
| `aggressive` | high tension |
| `danceable` | rhythmic drive |

### Derived emotion axes

#### Valence (pleasant ↔ unpleasant)

```text
positive  ←────────────→  negative
```

Computed from:
- happy (+)
- relaxed (+)
- sad (–)
- aggressive (–)

Range: **[-1 … +1]**

---

#### Arousal (calm ↔ excited)

```text
calm  ←────────────→  energetic
```

Computed from:
- aggressive (+)
- happy (+)
- relaxed (–)
- sad (–)

Range: **[-1 … +1]**

---

#### Intensity (drive / strength)

Derived mainly from:
- `danceable`

Range: **[0 … 1]**

This does **not** affect emotion meaning, only **strength of expression**.

---

## 3. Emotion → Color mapping (`emotion_color.py`)

This module is **authoritative**. Genre logic must never modify it directly.

### 3.1 Emotion quadrants

The emotion plane is divided into four semantic anchors:

| Valence | Arousal | Emotion |
|------|-------|--------|
| + | + | Joy |
| – | + | Anger |
| + | – | Calm |
| – | – | Sadness |

Each corner has a **configurable RGB color**.

---

### 3.2 Interpolation

Emotion color is interpolated bilinearly between corners based on valence & arousal.

This ensures:
- smooth transitions
- no sudden hue jumps

---

### 3.3 White center (neutral zone)

Near the center of the emotion plane, colors fade toward **white**.

Parameters:

```json
"white_center": {
  "color": [255, 255, 255],
  "radius": 0.25
}
```

- Radius is measured in **emotion space distance**
- Encourages calm / neutral lighting
- Prevents noisy flicker around neutral emotions

---

### 3.4 Intensity remapping

Low emotional intensity is boosted so visuals never go fully dark.

```text
0.0 ──┬──────────────┐
      │              │
      │   0.7        │
      └─────▶────────┘ 1.0
```

Parameter:

```json
"min_intensity": 0.7
```

This ensures:
- emotion is always visible
- intensity only adds expression

---

### 3.5 Special modes

#### PURE_COLOR

```json
"pure_color": true
```

- Disables interpolation
- Disables white center
- Each quadrant maps to a single fixed color

Useful for:
- symbolic visuals
- debugging
- very bold styles

---

#### MOOD_ONLY

```json
"mood_only": true
```

- Enables interpolation
- Disables white & intensity

Useful for:
- ambient / slow visuals
- non‑rhythmic environments

---

## 4. Genre Color Profiles

Genre profiles **do not change emotion meaning**.

They only control **how emotion is expressed** visually.

Each genre defines a stylistic baseline.

---

### 4.1 Core genre parameters

| Parameter | Meaning |
|--------|--------|
| `hue` | stylistic anchor hue |
| `sat_floor` | minimum saturation |
| `bright_floor` | minimum brightness |
| `mood_hue_weight` | how much emotion bends genre hue |
| `energy_boost` | amplifies motion & activity |
| `accent_gain` | beat / transient emphasis |
| `flash_decay` | how fast flashes fade |

These parameters mainly affect **motion & style**, not emotion semantics.

---

### 4.2 Emotion‑specific genre modifiers

These parameters *post‑shape* the emotion color output.

They never change valence or arousal.

---

#### `emotion_bright_gain`

```text
< 1.0 → darker, restrained
= 1.0 → neutral
> 1.0 → brighter, expressive
```

Used to adapt emotion visibility per genre.

Example:
- Jazz: subtle
- Pop: brighter

---

#### `emotion_sat_gain` (optional)

Controls how colorful emotions appear.

- Low values → muted, pastel
- High values → vivid, punchy

---

#### `emotion_white_gain`

Controls **how strongly low‑intensity emotion is pulled toward white**.

```text
low intensity emotion
     ↓
 calmer / neutral look
```

Typical use:
- Ambient genres → higher values
- Aggressive genres → lower values

---

#### `emotion_intensity_curve`

Shapes how emotional intensity grows.

| Curve | Behavior |
|-----|--------|
| `linear` | direct mapping |
| `soft` | emotion ramps up gently |
| `hard` | emotion stays calm until strong |

This affects **perceptual drama**.

---

## 5. WLED parameters

Emotion affects **both color and motion**.

---

### Speed

```python
speed = 40 + 200 * abs(arousal)
```

- Calm → slow motion
- High arousal → fast motion

---

### Effect intensity

```python
intensity_param = 50 + 200 * intensity
```

Controls:
- sparkle density
- wave strength
- strobe aggressiveness

---

## 6. Strobe logic (emotion‑driven)

Strobes are **not random**.

Triggered only when:
- arousal is high
- arousal changes suddenly
- emotion is tense or energetic

This avoids:
- fatigue
- constant flashing

---

## 7. Emotion Debug Visualization (OpenCV)

The debug window shows:

- Emotion plane (valence / arousal)
- White neutral zone
- Emotion quadrant labels
- Live emotion point

Extended debug adds:

- Base emotion color
- Genre‑shaped final color
- Intensity ring
- Genre modifier values

This tool is essential for tuning.

---

## 8. Practical tuning guidelines

### If visuals feel too chaotic
- Increase `emotion_white_gain`
- Reduce `emotion_bright_gain`
- Use `soft` intensity curve

### If visuals feel boring
- Increase `emotion_bright_gain`
- Use `hard` curve
- Reduce white radius

### If genre identity is lost
- Lower `mood_hue_weight`
- Increase `sat_floor`

---

## 9. Design principles (non‑negotiable)

- Emotion meaning must remain stable
- Genre must never override emotion
- White center prevents emotional noise
- Debug tools must mirror production logic

Breaking these rules will cause unpredictable visuals.

---

## 10. Summary

This system is:
- modular
- explainable
- tunable in real time

If something looks wrong, you can always identify **which layer** is responsible.

That is intentional.


---

## 11. Real‑world genre profile examples (JSON)

Below are **complete, realistic genre profiles** showing how emotion colors are *shaped*, not redefined. These are meant to be copied, tweaked, and compared.

---

### 11.1 Pop (bright, emotional, audience‑friendly)

```json
"Pop": {
  "hue": 320,
  "sat_floor": 0.54,
  "bright_floor": 0.48,

  "mood_hue_weight": 0.48,
  "energy_boost": 0.98,
  "accent_gain": 0.76,
  "flash_decay": 0.74,

  "emotion_bright_gain": 1.15,
  "emotion_sat_gain": 1.10,
  "emotion_white_gain": 0.85,
  "emotion_intensity_curve": "linear"
}
```

**Behavior**
- Emotions are **clearly visible**
- Whites are slightly suppressed → colors stay vivid
- Intensity grows predictably

Best for:
- mainstream pop
- vocal‑driven tracks
- stage & audience lighting

---

### 11.2 Techno / Electronic (driven, physical, restrained emotion)

```json
"Techno": {
  "hue": 200,
  "sat_floor": 0.64,
  "bright_floor": 0.45,

  "mood_hue_weight": 0.55,
  "energy_boost": 1.10,
  "accent_gain": 0.95,
  "flash_decay": 0.68,

  "emotion_bright_gain": 0.95,
  "emotion_sat_gain": 1.05,
  "emotion_white_gain": 0.60,
  "emotion_intensity_curve": "hard"
}
```

**Behavior**
- Emotion is present but **not dominant**
- Strong motion & rhythm response
- Neutral emotions quickly leave the white center

Best for:
- club environments
- rhythm‑centric visuals
- long‑form mixing

---

### 11.3 Ambient (subtle, emotional wash, low motion)

```json
"Ambient": {
  "hue": 240,
  "sat_floor": 0.38,
  "bright_floor": 0.32,

  "mood_hue_weight": 0.35,
  "energy_boost": 0.70,
  "accent_gain": 0.40,
  "flash_decay": 0.85,

  "emotion_bright_gain": 0.80,
  "emotion_sat_gain": 0.75,
  "emotion_white_gain": 1.25,
  "emotion_intensity_curve": "soft"
}
```

**Behavior**
- Emotion gently fades toward white
- Low arousal stays calm and spacious
- No sudden visual jumps

Best for:
- installations
- background environments
- long ambient pieces

---

### 11.4 Reading these examples correctly

Key reminder:

> **Emotion decides the color direction**  
> **Genre decides how loud that color speaks**

If two genres receive the *same* emotion input:
- they will point to the **same emotional color**
- but differ in brightness, saturation, motion, and decay

This is the core design invariant of the system.


---

## 12. How to design your own genre (5-step recipe — COMPLETE)

This is a **strict, complete, copy-safe recipe**.
If you follow these 5 steps, you will produce a genre profile that:
- loads without errors
- respects emotion semantics
- behaves predictably in WLED

No theory, no psychology — only actions.

---

### Step 1 — Start from DEFAULT (never from scratch)

Always copy the DEFAULT profile first.
This guarantees stability.

```json
"My Genre": {
  "hue": 210,
  "sat_floor": 0.30,
  "bright_floor": 0.30,
  "mood_hue_weight": 0.45,
  "energy_boost": 1.00,
  "accent_gain": 1.00,
  "flash_decay": 0.88
}
```

Do **not** remove keys. Only modify values.

---

### Step 2 — Choose the genre identity color

**Action:** Set `hue` only.

Rules:
- 0–40   → aggressive / physical
- 40–120 → warm / organic
- 160–220 → calm / technical
- 260–330 → emotional / cinematic

```json
"hue": 35
```

Test immediately:
- Neutral emotion must already look acceptable.

---

### Step 3 — Lock minimum visibility (anti-flicker step)

**Action:** Set the two floors.

Hard limits (do not cross):
- `sat_floor`   ≥ 0.25
- `bright_floor` ≥ 0.25

Guidelines:
- Club / EDM → 0.45–0.70
- Pop / Rock → 0.40–0.55
- Ambient / Classical → 0.30–0.40

```json
"sat_floor": 0.55,
"bright_floor": 0.45
```

If visuals flicker or disappear → raise these values.

---

### Step 4 — Decide emotion vs genre dominance

**Action:** Tune `mood_hue_weight`.

Meaning:
- 0.25 → genre color dominates
- 0.45 → balanced (recommended)
- 0.60 → emotion dominates

```json
"mood_hue_weight": 0.48
```

Validation:
- Joy / Anger / Calm / Sadness must still be recognizable
- Genre color must still be identifiable

---

### Step 5 — Shape motion & transients (energy layer)

These parameters **do not affect color meaning**.
They control movement and punch.

Safe operating ranges:
- `energy_boost` : 0.80 – 1.20
- `accent_gain`  : 0.50 – 1.00
- `flash_decay`  : 0.65 – 0.90

```json
"energy_boost": 1.05,
"accent_gain": 0.80,
"flash_decay": 0.72
```

Symptoms:
- Too nervous → raise `flash_decay`
- Too flat → raise `energy_boost`

---

### Optional Step — Emotion expression modifiers (advanced)

Only use these **after** the genre works.

```json
"emotion_bright_gain": 1.00,
"emotion_sat_gain": 1.00,
"emotion_white_gain": 1.00,
"emotion_intensity_curve": "linear"
```

Valid curves:
- `linear` (default)
- `soft`   (ambient / background)
- `hard`   (EDM / drops)

---

### Fully expanded valid genre example

```json
"Custom Genre": {
  "hue": 35,
  "sat_floor": 0.55,
  "bright_floor": 0.45,
  "mood_hue_weight": 0.48,
  "energy_boost": 1.05,
  "accent_gain": 0.80,
  "flash_decay": 0.72,

  "emotion_bright_gain": 1.00,
  "emotion_sat_gain": 1.00,
  "emotion_white_gain": 1.00,
  "emotion_intensity_curve": "linear"
}
```

---

### Final sanity checklist (mandatory)

Before committing a genre:
- Neutral emotion → calm, readable, no flicker
- High arousal → faster motion, not color chaos
- Calm emotion → stable, not washed out
- Genre is recognizable even when emotion changes

If this fails:
- Fix the **genre profile**
- Do **not** modify emotion math or emotion colors

**Emotion defines meaning. Genre defines expression.**

