# osc_schema.py
import json

class OscModelSchema:
    def __init__(self, model_json, base_path='/'):
        with open(model_json, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.model_name = self._slug(meta.get("name", "model"))
        self.labels = meta.get("classes", meta.get("labels", []))

        self.paths = {
            label: f"/{self.model_name}/{self._label_to_path(label)}"
            for label in self.labels
        }

    def _slug(self, s: str) -> str:
        return (
            s.lower()
             .replace(" ", "_")
             .replace("&", "and")
             .replace("/", "_")
        )

    def _label_to_path(self, label: str) -> str:
        return (
            label.lower()
                 .replace(" ", "_")
                 .replace("&", "and")
                 .replace("---", "/")
                 .replace("/", "/")
        )
