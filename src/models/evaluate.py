import argparse
import joblib
from sklearn.metrics import classification_report
from tensorflow import keras
from src.data.build_features import make_split_for_target
from src.models.common import df_to_inputs
from src.utils.config import MODELS_DIR

def main(task: str, model_name: str):
    target_col = "priority_label" if task == "priority" else "routing_label"
    _, X_test, _, y_test = make_split_for_target(target_col)
    model = keras.models.load_model(MODELS_DIR / f"{task}_{model_name}.keras")
    encoder = joblib.load(MODELS_DIR / f"{target_col}_encoder.pkl")
    y_pred = model.predict(df_to_inputs(X_test), verbose=0).argmax(axis=1)
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["priority", "routing"], required=True)
    parser.add_argument("--model", choices=["custom", "pretrained"], default="custom")
    args = parser.parse_args()
    main(args.task, args.model)
