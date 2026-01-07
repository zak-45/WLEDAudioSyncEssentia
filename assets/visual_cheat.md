---

# 🎨 Audio → Color System

## Visual Cheat Sheet

---

## 🧠 SIGNAL FLOW (Mental Model)

```
Audio
 ├─ RMS / Beat ─────────────┐
 ├─ Genre Classifier ──┐    │
 ├─ AUX (Danceability) │    │
 └─ Mood Analysis ─────┴────┘
            ↓
      Energy + Valence
            ↓
      Genre Hue + Mood Hue
            ↓
     Circular Hue Fusion
            ↓
  🎨 Final Color + Accent Color
```

---

## 🎵 GENRE → BASE COLOR (HUE)

| Genre       | Hue         | Visual        |
| ----------- | ----------- | ------------- |
| Rock        | 🔴 10–20°   | Red / Power   |
| Electronic  | 🟣 270–290° | Neon Purple   |
| Hip Hop     | 🟠 25°      | Gold          |
| Funk / Soul | 🟡 30°      | Groove Orange |
| Pop         | 💗 330°     | Pink          |
| Jazz        | 🔵 200°     | Cool Blue     |
| Classical   | 🟢 140°     | Calm Green    |
| Reggae      | 🟡🟢 90°    | Sunny Green   |
| Blues       | 🔵 220°     | Deep Blue     |

*(Hue never disappears — mood only bends it)*

---

## 😊 MOOD SPACE (VALENCE × ENERGY)

```
          ENERGY ↑
                |
    Aggressive   |   Euphoric
     (Red)       |    (Pink)
                |
  ---------------+--------------→ VALENCE
                |
      Dark       |     Calm
    (Blue)       |   (Green)
                |
```

* **Valence** → left/right (sad → happy)
* **Energy** → bottom/top (calm → intense)

---

## 🎚 PARAMETER EFFECTS

```
Hue        → WHAT color
Saturation → HOW strong
Brightness → HOW loud
```

| Parameter    | Visual Result     |
| ------------ | ----------------- |
| ↑ Saturation | More vivid        |
| ↓ Saturation | Pastel / muted    |
| ↑ Brightness | Loud / aggressive |
| ↓ Brightness | Soft / ambient    |

---

## ⚡ BEAT & ACCENT LOGIC

```
Beat detected?
 ├─ Yes → Accent = 1.0
 └─ No  → Accent *= Decay
```

### Decay feels like:

| Genre      | Flash Shape     |
| ---------- | --------------- |
| Metal      | Sharp spike ⚡   |
| Rock       | Punchy hit 🔥   |
| Electronic | Smooth pulse 🌊 |
| Jazz       | Soft glow ✨     |
| Classical  | Slow wave 🌙    |

---

## 🎭 FINAL COLOR LAYERS

```
[ Genre Color ]  ← identity
        +
[ Mood Color ]   ← emotion
        +
[ Accent Color ] ← rhythm
```

Accent color is usually:

* Complementary
* Brighter
* Short-lived

---

## 🌍 PRESET PERSONALITIES

### 🔥 Club

```
High Saturation
Fast Flash
Genre Dominant
```

### 🏛 Installation

```
Balanced
Slow Evolution
Mood Dominant
```

### 🌿 Ambient

```
Low Brightness
Soft Colors
Almost No Flash
```

---

## 🧪 DEBUG QUICK READ

```
GENRE TOP5 → identity
MACRO CONF → confidence
Genre color → base truth
Mood color → emotional tint
Final color → production output
Accent color → beat energy
```

---

## 🧩 ONE-LINE SUMMARY

> **Genre sets the color, mood bends it, beats make it breathe.**

---
