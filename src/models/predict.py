import joblib
import pandas as pd
from tensorflow import keras
from src.models.common import df_to_inputs
from src.utils.config import MODELS_DIR

def predict_ticket(sample: dict, task: str = "priority", model_name: str = "custom"):
    target_col = "priority_label" if task == "priority" else "routing_label"
    model = keras.models.load_model(MODELS_DIR / f"{task}_{model_name}.keras")
    encoder = joblib.load(MODELS_DIR / f"{target_col}_encoder.pkl")
    df = pd.DataFrame([sample])
    y_pred = model.predict(df_to_inputs(df), verbose=0).argmax(axis=1)[0]
    return encoder.inverse_transform([y_pred])[0]
