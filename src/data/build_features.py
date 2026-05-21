import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.utils.config import PROCESSED_DATA_PATH, MODELS_DIR, RANDOM_STATE

def load_processed_data():
    return pd.read_csv(PROCESSED_DATA_PATH)

def make_split_for_target(target_col: str):
    df = load_processed_data()
    X = df.drop(columns=["priority_label", "routing_label"])
    y = df[target_col].astype(str)
    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoder, MODELS_DIR / f"{target_col}_encoder.pkl")
    return train_test_split(X, y_enc, test_size=0.2, stratify=y_enc, random_state=RANDOM_STATE)
