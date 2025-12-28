import numpy as np
from essentia.standard import TensorflowPredictEffnetDiscogs, TensorflowPredict2D

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
        self.labels = self.classifier.outputNames()

    def classify(self, audio_chunk):
        audio_chunk = np.asarray(audio_chunk, dtype=np.float32)
        embeddings = self.effnet(audio_chunk)

        if len(embeddings) == 0:
            return None

        return self.classifier(np.asarray(embeddings, dtype=np.float32))
