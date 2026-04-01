import numpy as np
import pickle
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import confusion_matrix
from sktime.transformations.panel.rocket import Rocket

# --------------------------------------
# Re-define the same class structure
# Required for pickle loading if saved this way
# --------------------------------------
def normalize_samples(X):
    mean = X.mean(axis=2, keepdims=True)
    std = X.std(axis=2, keepdims=True)
    std[std == 0] = 1.0
    return (X - mean) / std

class OriginalRocketClassifier:
    def __init__(self, C=1.0, max_iter=1000, normalize=False, augment_n=0):
        self.normalize = normalize
        self.augment_n = augment_n
        self.transformer_ = Rocket(random_state=42)
        self.clf_ = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        self.label_encoder_ = None
        self.class_names_ = None

    def predict(self, X):
        if self.normalize:
            X = normalize_samples(X)
        FX = self.transformer_.transform(X)
        return self.clf_.predict(FX)

    def predict_label_names(self, X):
        preds = self.predict(X)
        if self.class_names_ is None:
            return preds
        return [self.class_names_[int(p)] for p in preds]

# --------------------------------------
# Load and run inference
# --------------------------------------
if __name__ == "__main__":
    model_path = "original_rocket_baseline.pkl"

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print("Model loaded successfully.")
    print("Inference device: CPU")

    # Example single sample
    # Replace with your real test sample of shape [1, n_channels, time_length]
    X_test = np.random.randn(1, 52, 500).astype(np.float32)

    pred_index = model.predict(X_test)[0]
    pred_label = model.predict_label_names(X_test)[0]

    print("Predicted class index:", pred_index)
    print("Predicted label:", pred_label)