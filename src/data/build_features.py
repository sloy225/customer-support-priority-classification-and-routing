import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.utils.config import PROCESSED_DATA_PATH, MODELS_DIR, RANDOM_STATE


def load_processed_data():
    return pd.read_csv(PROCESSED_DATA_PATH)


def make_split_for_target(target_col: str):
    df = load_processed_data()

    X = df.drop(columns=["priority_label", "routing_label"])
    y = df[target_col].astype(str)

    # 1. SPLIT AVANT ENCODING (IMPORTANT)
    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # 2. LABEL ENCODER FIT UNIQUEMENT SUR TRAIN
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_raw)
    y_test = encoder.transform(y_test_raw)

    # 3. SAUVEGARDE ENCODER
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoder, MODELS_DIR / f"{target_col}_encoder.pkl")

    print("Classes :", encoder.classes_)
    print("Nb classes :", len(encoder.classes_))

    return X_train, X_test, y_train, y_test