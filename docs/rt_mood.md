
---

# RTMood Pipeline Documentation

## Purpose

RTMood provides a **perceptual emotional color mapping** derived from **valence** and **energy**, designed to reflect *emotional direction* rather than raw brightness or LED intensity.

RTMood is **not** a lighting model.
It is an **emotion-space renderer**.

---

## Design Principles

1. **Semantic correctness over brightness**
2. **Stable emotional geometry**
3. **Protection against chroma collapse**
4. **Isolation from physical energy scaling**
5. **Optional post-mapping artistic lifting**

---

## Inputs

| Parameter         | Range        | Meaning              |
| ----------------- | ------------ | -------------------- |
| `valence`         | `[0.0, 1.0]` | Emotional positivity |
| `activity_energy` | `[0.0, 1.0]` | Physical intensity   |

These values come from upstream analysis and are **not modified destructively** before RTMood.

---

## Step 1 — Valence Expansion (Perceptual Contrast)

### Problem

Valence is intentionally conservative (~0.4–0.6).
RTMood expects **stronger emotional contrast**.

### Solution

Non-linear expansion around neutral (0.5):

```python
v = float(valence)

v_expanded = (
    0.5 +
    np.sign(v - 0.5) *
    abs(v - 0.5) ** 0.6
)

v_expanded = np.clip(v_expanded, 0.0, 1.0)
```

### Effect

* Neutral stays neutral
* Extremes expand perceptually
* No clipping or jumps

---

## Step 2 — RTMood Valence Mapping

RTMood expects valence in `[-1, +1]`:

```python
soft_valence = (v_expanded * 2.0) - 1.0
```

| Valence | Meaning                 |
| ------- | ----------------------- |
| `-1.0`  | Strong negative emotion |
| `0.0`   | Neutral                 |
| `+1.0`  | Strong positive emotion |

---

## Step 3 — Energy Mapping (Critical Protection)

### Problem

RTMood **collapses chroma near ±1 energy**.

### Correct Mapping

```python
soft_energy = (activity_energy * 2.0) - 1.0
```

### Perceptual Clamp (Mandatory)

```python
SOFT_E_MAX = 0.75
SOFT_E_MIN = -0.75

soft_energy = np.clip(
    soft_energy,
    SOFT_E_MIN,
    SOFT_E_MAX
)
```

### Why This Is Required

| Without clamp | With clamp           |
| ------------- | -------------------- |
| Blackouts     | Stable chroma        |
| Washed colors | Emotional coherence  |
| Unstable LEDs | Predictable behavior |

---

## Step 4 — RTMood Color Lookup

```python
r, g, b = rt_color_mapper.get_rgb(
    soft_valence,
    soft_energy
)
```

### Output Characteristics

* Color reflects **emotion**, not light power
* High energy ≠ bright
* High energy = tension / compression
* Calm + positive = lighter pastels

This is **intentional RTMood behavior**.

---

## Step 5 — Output Handling

RTMood output is treated as a **semantic mood layer**:

```python
osc.send("/WASEssentia/mood/rt/color/*", rgb / 255.0)
```

It is **not fused** with genre or final production colors by default.

---

## Optional — LED / Visual Lift Layer (Recommended for Shows)

RTMood is emotionally correct but may appear dark on LEDs.
A **post-mapping lift** preserves semantics:

```python
rt_lift = 0.4 + 0.6 * activity_energy

r = min(255, int(r * rt_lift))
g = min(255, int(g * rt_lift))
b = min(255, int(b * rt_lift))
```

### Important

This **must not** affect:

* `soft_valence`
* `soft_energy`
* RTMood internal mapping

---

## What RTMood Is NOT

❌ A brightness controller
❌ A saturation maximizer
❌ A genre mapper
❌ A beat-reactive system

RTMood answers one question only:

> **“What emotional space does this music inhabit right now?”**

---

## Summary

| Stage             | Role               |
| ----------------- | ------------------ |
| Valence expansion | Emotional contrast |
| Energy clamp      | Stability          |
| RTMood mapping    | Emotional color    |
| Optional lift     | Display adaptation |

---

## Final Notes

* The current implementation is **correct**
* No further fixes are required
* Any further changes are **artistic decisions**

---
