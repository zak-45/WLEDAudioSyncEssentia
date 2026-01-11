Below is a **README-level explanation**, written for **users, artists, and integrators**, not just developers.
It explains *what RTMood is*, *why it exists*, and *how to interpret it* — without leaking internal math.

---

# RTMood — Real-Time Emotional Color Mapping

RTMood is a **real-time emotional color system** driven by music analysis.

It converts **musical emotion** into **color**, focusing on *how the music feels*, not how loud or bright it is.

RTMood runs alongside genre-based and production color pipelines, but it serves a **different purpose**:
to represent **emotional direction and intensity** in a perceptually meaningful way.

---

## What RTMood Represents

RTMood uses two emotional dimensions:

| Dimension   | Meaning                                       |
| ----------- | --------------------------------------------- |
| **Valence** | How positive or negative the emotion feels    |
| **Energy**  | How tense, intense, or calm the emotion feels |

These are mapped into a continuous emotional color space.

RTMood answers the question:

> *“What emotional space does the music occupy right now?”*

---

## What RTMood Is (and Is Not)

### RTMood **is**

* An emotional visualization layer
* Stable and smooth over time
* Independent from lighting brightness
* Designed for emotional coherence

### RTMood **is NOT**

* A beat-reactive lighting effect
* A brightness or saturation controller
* A genre color system
* A replacement for production colors

RTMood colors may appear **subtle or muted** — this is intentional.

---

## Emotional Mapping Overview

RTMood maps music into a 2D emotional space:

### Valence (horizontal axis)

* Negative → melancholic, dark, tense
* Neutral → balanced, stable
* Positive → joyful, warm, open

### Energy (vertical axis)

* Low → calm, soft, relaxed
* High → tense, compressed, aggressive

The resulting color reflects **emotional character**, not volume or rhythm.

---

## Why RTMood Colors Can Look Dark at High Energy

High energy does **not** mean bright or colorful.

In emotional terms:

* Very high energy often feels **tense**, **compressed**, or **aggressive**
* RTMood reflects this by **reducing chroma**
* This prevents emotional misrepresentation

If you want brighter visuals, RTMood should be **post-processed**, not altered internally.

---

## RTMood vs Final Production Color

| Feature            | RTMood | Final Color |
| ------------------ | ------ | ----------- |
| Emotional accuracy | ✅      | ⚠️          |
| Genre identity     | ❌      | ✅           |
| LED readiness      | ⚠️     | ✅           |
| Artistic styling   | ❌      | ✅           |

RTMood is **truthful**.
Final colors are **expressive**.

They are designed to coexist.

---

## Recommended Usage

### Best Use Cases

* Emotional overlays
* Ambient installations
* Mood indicators
* Visual debugging
* Emotional timelines
* Cross-media emotion sync

### Not Recommended

* Direct stage lighting without post-processing
* Beat-driven effects
* High-contrast strobes

---

## Making RTMood Brighter (Optional)

If RTMood looks too dark on LEDs, apply a **visual lift** *after* mapping:

```text
RTMood → (optional brightness lift) → display
```

This preserves emotional correctness while adapting to hardware.

---

## Stability & Safety

RTMood is protected against:

* Color collapse
* Extreme saturation
* Energy-related blackouts
* Sudden emotional jumps

It is safe to run continuously in real-time systems.

---

## Philosophy

RTMood is not about spectacle.

It is about **emotional truth**.

Where other systems ask:

> *“How should this look?”*

RTMood asks:

> *“How does this feel?”*

---

