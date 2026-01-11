"""Helpers for loading genre label metadata from model JSON files.

This module reads the label list associated with a trained genre classifier
so that raw prediction vectors can be mapped to human-readable genre names.
It keeps the label ordering consistent with the model metadata used by
Essentia and the TensorFlow graphs.
"""

import json

def load_genre_labels(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # This matches Essentia.js structure
    labels = data["classes"]

    return labels
