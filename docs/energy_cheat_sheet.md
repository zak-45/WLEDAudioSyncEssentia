# 🎛️ Energy Cheat Sheet (One‑Page)

This is a **fast reference** to understand, debug, and tune energy in the system.

---

## 1️⃣ Two Energies — Never Confuse Them

| Name | What it means | What it controls |
|----|----|----|
| **activity_energy** | Physical motion in sound | Brightness, flash strength, motion |
| **emotion_energy** | Emotional intensity | Color tension, saturation, mood depth |

> ⚠️ Loud ≠ Active ≠ Emotional

---

## 2️⃣ activity_energy (Physical Motion)

**Answers:** *How much is the sound moving right now?*

### Ingredients

- **RMS** → *Gate only* (silence detection)
- **Motion** → RMS of first derivative (micro‑movement)
- **ZCR** → Transient density (percussiveness)

### Formula

```
raw_activity =
    0.7 × motion_norm
  + 0.3 × zcr_norm

activity_energy = smooth(raw_activity)
```

### Expected Values

| Music | activity_energy |
|----|----|
| Ambient pad | 0.05 – 0.15 |
| Johnny Cash – Hurt (intro) | 0.15 – 0.30 |
| Blues / Rock ballad | 0.30 – 0.45 |
| Classic Rock | 0.45 – 0.65 |
| Punk / Metal | 0.65 – 0.90 |

### Tuning Knobs

- Increase **motion weight** → more sensitivity to distortion
- Increase **ZCR weight** → more sensitivity to drums
- Increase **smoothing α** → calmer visuals

---

## 3️⃣ emotion_energy (Emotional Intensity)

**Answers:** *How intense does this feel emotionally?*

### Inputs

| Signal | Meaning |
|----|----|
| activity_energy | Physical drive |
| valence | Positive ↔ Negative emotion |
| genre confidence | Certainty / focus |

### Formula

```
emotion_energy =
    0.45 × activity_energy
  + 0.35 × (1 − valence)
  + 0.20 × genre_confidence
```

### Interpretation

| Scenario | Result |
|----|----|
| Quiet but sad | Medium emotion_energy |
| Loud but happy | Medium emotion_energy |
| Loud + angry | High emotion_energy |
| Calm + neutral | Low emotion_energy |

---

## 4️⃣ How Energy Drives Visuals

| Parameter | Driven by |
|----|----|
| Brightness | activity_energy |
| Flash decay | activity_energy |
| Accent strength | activity_energy × beat |
| Mood saturation | emotion_energy |
| Hue tension | emotion_energy + valence |

---

## 5️⃣ Debug Checklist

### If **everything feels the same**

- RMS used as energy ❌
- Too much smoothing ❌
- Motion normalization too narrow ❌

### If **Hurt looks like Rock**

- emotion_energy over‑weighted by activity ❌
- valence weight too low ❌

### If **Rock feels weak**

- motion_ref too high
- ZCR underweighted

---

## 6️⃣ Golden Rules

- ❌ Never use RMS as energy
- ✅ Motion = activity
- ✅ Valence modulates emotion
- ✅ Silence resets history
- 🎵 Energy must feel right — not measure right

---

## 7️⃣ Mental Model

> **activity_energy** = *body movement*  
> **emotion_energy** = *heart pressure*

Both are needed. Neither alone is enough.

---

✅ If this page makes sense, your system is healthy.

