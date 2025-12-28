from collections import defaultdict

def build_macro_mapping(labels):
    """
    Builds a mapping:
    macro_genre -> list of label indices
    """
    macro_map = defaultdict(list)

    for i, label in enumerate(labels):
        macro = label.split('---')[0].strip()
        macro_map[macro].append(i)

    return dict(macro_map)


def collapse_to_macro(probs, labels, agg="mean"):
    """
    Collapse 400-dim probs into macro genres.
    agg: "mean" or "max"
    """
    macro_map = build_macro_mapping(labels)
    macro_probs = {}

    for macro, idxs in macro_map.items():
        values = probs[idxs]
        if agg == "max":
            macro_probs[macro] = float(values.max())
        else:
            macro_probs[macro] = float(values.mean())

    return macro_probs
