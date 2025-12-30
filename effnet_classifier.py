import json
import numpy as np
from essentia.standard import TensorflowPredictEffnetDiscogs, TensorflowPredict2D
from labels import load_genre_labels

class EffnetClassifier:
    def __init__(self, model_dir):
        self.effnet = TensorflowPredictEffnetDiscogs(
            graphFilename=f"{model_dir}/discogs-effnet-bs64-1.pb",
            output="PartitionedCall:1"
        )

        self.classifier = TensorflowPredict2D(
            graphFilename=f"{model_dir}/genre_discogs400-discogs-effnet-1.pb",
            input="serving_default_model_Placeholder",
            output="PartitionedCall:0"
        )

        # LOAD LABELS FROM JSON (THIS IS THE KEY FIX)
        self.labels = load_genre_labels(
            f"{model_dir}/genre_discogs400-discogs-effnet-1.json"
        )

        print("Number of Labels loaded:", len(self.labels))

    def classify(self, audio):
        audio = np.asarray(audio, dtype=np.float32)

        embeddings = self.effnet(audio)
        if len(embeddings) == 0:
            return None

        preds = self.classifier(np.asarray(embeddings, dtype=np.float32))

        # IMPORTANT: average over time axis
        probs = preds.mean(axis=0)

        if len(probs) != len(self.labels):
            print("❌ size mismatch:", len(probs), len(self.labels))
            return None

        return probs

    def compute_embeddings(self, audio):
        audio = np.asarray(audio, dtype=np.float32)
        return self.effnet(audio)


class AuxClassifier:
    def __init__(self, model_pb, model_json, agg="mean"):
        self.agg = agg

        with open(model_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.labels = meta.get("classes", meta.get("labels"))

        # Use TensorflowPredict2D to avoid 'cppPool' errors with TensorflowPredict and numpy arrays
        self.model = TensorflowPredict2D(
            graphFilename=model_pb,
            input="model/Placeholder",
            output="model/Softmax"
        )

    def classify(self, embeddings):
        """
        embeddings: np.ndarray (frames, 1280)
        returns: dict {label: prob}
        """
        if embeddings is None or len(embeddings) == 0:
            return None

        embeddings = np.asarray(embeddings)

        # Pool embeddings over time
        if embeddings.ndim == 2:
            if self.agg == "max":
                pooled = embeddings.max(axis=0)
            else:
                pooled = embeddings.mean(axis=0)
        """
        if embeddings.ndim == 2:
            pooled = embeddings.mean(axis=0)  # (1280,)
        else:
            pooled = embeddings               # already (1280,)
        """

        pooled = pooled.astype(np.float32)
        # Reshape to (1, 1280) for TensorflowPredict2D
        pooled = pooled.reshape(1, -1)

        preds = self.model(pooled)
        preds = np.asarray(preds).flatten()
        # Not strictly required, but safe for OSC consumers.
        preds = np.clip(preds, 0.0, 1.0)

        return dict(zip(self.labels, preds))
