import json

def load_genre_labels(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # This matches Essentia.js structure
    labels = data["classes"]

    return labels
