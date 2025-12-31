import os
import json

def discover_models(models_dir):
    """
    Finds (.pb, .json) model pairs in models_dir
    Returns list of dicts:
      {
        "name": str,
        "pb": str,
        "json": str,
        "type": "genre" | "aux",
        "output_name": str,
      }
    Exclude : discogs-effnet-bs64-1.pb
    """

    models = []

    if not os.path.isdir(models_dir):
        raise RuntimeError(f"Models folder not found: {models_dir}")

    files = os.listdir(models_dir)
    files.remove('discogs-effnet-bs64-1.pb')

    pbs = {f[:-3]: f for f in files if f.endswith(".pb")}
    jsons = {f[:-5]: f for f in files if f.endswith(".json")}

    for key in sorted(pbs.keys() & jsons.keys()):
        json_path = os.path.join(models_dir, jsons[key])

        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        schema = meta.get("schema")
        output_name = schema.get('outputs')[0]['name']

        model_type = "genre" if 'genre' in key else "aux"

        models.append({
            "name": meta.get("name", key),
            "pb": os.path.join(models_dir, pbs[key]),
            "json": json_path,
            "type": model_type,
            "output_name": output_name,
        })

    if not models:
        raise RuntimeError("No valid models found in models folder")

    return models
