import argparse
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

from src.data.clean_data import prepare_dataset
from src.data.build_features import make_split_for_target
from src.models.model_custom import build_custom_model
from src.models.model_pretrained import build_pretrained_style_model
from src.models.common import df_to_inputs
from src.utils.config import MODELS_DIR, CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS


def compute_tabular_metadata(X_train):
    categorical_vocabularies = {
        col: sorted(X_train[col].astype(str).unique().tolist())
        for col in CATEGORICAL_COLUMNS
    }
    numerical_means = {
        col: float(np.mean(X_train[col].astype("float32").values))
        for col in NUMERICAL_COLUMNS
    }
    numerical_vars = {
        col: float(np.var(X_train[col].astype("float32").values))
        for col in NUMERICAL_COLUMNS
    }
    return categorical_vocabularies, numerical_means, numerical_vars


def main(model_name: str):
    prepare_dataset()

    X_train, X_test, y_train, y_test = make_split_for_target("priority_label")

    # Encodage de la cible
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # Sauvegarde de l'encoder pour evaluate.py et predict.py
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(le, MODELS_DIR / "priority_label_encoder.pkl")

    train_inputs = df_to_inputs(X_train)
    test_inputs = df_to_inputs(X_test)

    categorical_vocabularies, numerical_means, numerical_vars = compute_tabular_metadata(X_train)

    if model_name == "custom":
        model, vectorizer = build_custom_model(
            num_classes=len(le.classes_),
            categorical_vocabularies=categorical_vocabularies,
            numerical_means=numerical_means,
            numerical_vars=numerical_vars
        )
    else:
        model, vectorizer = build_pretrained_style_model(
            num_classes=len(le.classes_),
            categorical_vocabularies=categorical_vocabularies,
            numerical_means=numerical_means,
            numerical_vars=numerical_vars
        )

    vectorizer.adapt(X_train["text"].astype(str).tolist())

    model.fit(
        train_inputs, y_train_enc,
        validation_split=0.2,
        epochs=5,
        batch_size=32,
        verbose=1
    )

    model.evaluate(test_inputs, y_test_enc, verbose=1)
    model.save(MODELS_DIR / f"priority_{model_name}.keras")
    print(f"Modèle sauvegardé : priority_{model_name}.keras")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["custom", "pretrained"], default="custom")
    args = parser.parse_args()
    main(args.model)