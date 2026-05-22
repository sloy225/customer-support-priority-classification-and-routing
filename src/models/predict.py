import numpy as np
import joblib
from tensorflow import keras
from src.models.common import df_to_inputs
from src.utils.config import MODELS_DIR
import pandas as pd

# Cache des modèles (chargés une seule fois)
_models = {}

def _load_model(task: str, model_name: str):
    key = f"{task}_{model_name}"
    if key not in _models:
        path = MODELS_DIR / f"{task}_{model_name}.keras"
        _models[key] = keras.models.load_model(path)
    return _models[key]


def predict_ticket(sample: dict, task: str, model_name: str = "custom") -> str:
    model = _load_model(task, model_name)

    # Charge les noms de classes
    class_names = joblib.load(MODELS_DIR / f"{task}_class_names.pkl")

    # Convertit le sample en DataFrame puis en inputs Keras
    df = pd.DataFrame([sample])
    inputs = df_to_inputs(df)

    # Prédiction
    probs = model.predict(inputs, verbose=0)
    idx   = int(np.argmax(probs, axis=1)[0])
    return class_names[idx]