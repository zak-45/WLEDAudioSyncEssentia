---

# 🎨 Genre Color Profiles – README

This file explains how **genre color profiles** work and how to tune them to shape the visual behavior of music-driven lighting.

Each genre has a profile that controls **color identity**, **energy response**, and **beat accent behavior**.

Profiles are stored as JSON and loaded at runtime.

---

## 🧠 Core Idea

Each genre has:

* a **semantic color identity** (hue)
* a **base color strength**
* a **reaction to musical energy**
* a **beat accent personality**
* a **balance between genre identity and mood**

This lets Rock feel *punchy*, Jazz feel *smooth*, and Classical feel *soft* — even at the same volume.

---

## 🧩 Profile Structure

Example:

```json
"Rock": {
  "hue": 0,
  "sat_floor": 0.73,
  "bright_floor": 0.69,
  "energy_boost": 0.88,
  "accent_gain": 0.90,
  "flash_decay": 0.69,
  "mood_hue_weight": 0.55
}
```

---

## 🎯 Parameter Reference

### 🎨 `hue` (0–360)

**What it does:**
Defines the *base color identity* of the genre.

* Red = aggression / power
* Yellow = warmth / groove
* Blue = calm / depth
* Purple = cinematic / synthetic

**Tuning tips:**

* Change only if the genre’s “color meaning” feels wrong
* Keep consistent across the system

---

### 🌈 `sat_floor` (0–1)

**What it does:**
Minimum color saturation — how *colorful* the genre always is.

| Value | Effect              |
| ----- | ------------------- |
| 0.3   | almost gray         |
| 0.6   | rich but controlled |
| 0.8   | very vivid          |

**Tuning tips:**

* Raise for expressive genres (Rock, Electronic)
* Lower for subtle genres (Jazz, Classical)

---

### 💡 `bright_floor` (0–1)

**What it does:**
Minimum brightness level, even when energy is low.

| Value | Effect          |
| ----- | --------------- |
| 0.3   | dark, moody     |
| 0.6   | clearly visible |
| 0.8   | always bright   |

**Tuning tips:**

* Lower for emotional depth
* Raise if lights feel “dead” at low energy

---

### ⚡ `energy_boost` (0–1)

**What it does:**
How strongly musical **energy** affects brightness.

* High → dramatic light response
* Low → stable, smooth visuals

**Tuning tips:**

* Increase for Rock / Metal / EDM
* Reduce for Ambient / Jazz / Classical

---

### ✨ `accent_gain` (0–1)

**What it does:**
How strong beat accents (flashes) are.

| Value | Effect         |
| ----- | -------------- |
| 0.3   | subtle pulse   |
| 0.6   | visible accent |
| 0.9   | punchy flash   |

**Tuning tips:**

* Increase for rhythm-driven genres
* Reduce for flowing music

---

### 🫀 `flash_decay` (0–1)

**What it does:**
How fast beat flashes fade.

| Value | Effect        |
| ----- | ------------- |
| 0.65  | sharp, punchy |
| 0.75  | balanced      |
| 0.85  | long glow     |

**Tuning tips:**

* Lower = strobe-like impact
* Higher = breathing / cinematic feel

---

### 🎭 `mood_hue_weight` (0–1)

**What it does:**
Balance between **genre color** and **mood color**.

* `0.25` → genre dominates
* `0.50` → balanced
* `0.65+` → mood dominates

**Tuning tips:**

* Raise for emotional genres
* Lower for identity-driven genres

---

## 🧪 Practical Tuning Examples

### Rock feels too yellow?

→ Lower `mood_hue_weight`

### Lights feel too flat?

→ Increase `energy_boost`

### Beat flashes are annoying?

→ Increase `flash_decay` or reduce `accent_gain`

### Jazz feels too aggressive?

→ Lower `sat_floor` and `accent_gain`

---

## 🧠 Design Philosophy

* **Genre defines identity**
* **Mood defines emotion**
* **Energy defines motion**
* **Beats define punctuation**

Each parameter adjusts *one dimension* only — this prevents chaos and keeps visuals musical.

---

## ✅ Best Practices

* Change **one parameter at a time**
* Test with real music
* Trust your eyes more than numbers
* Keep profiles consistent across genres

---
