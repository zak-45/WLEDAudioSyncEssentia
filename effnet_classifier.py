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
