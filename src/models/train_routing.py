import argparse
from src.data.clean_data import prepare_dataset
from src.data.build_features import make_split_for_target
from src.models.model_custom import build_custom_model
from src.models.model_pretrained import build_pretrained_style_model
from src.models.common import df_to_inputs
from src.utils.config import MODELS_DIR

def main(model_name: str):
    prepare_dataset()
    X_train, X_test, y_train, y_test = make_split_for_target("routing_label")
    train_inputs = df_to_inputs(X_train)
    test_inputs = df_to_inputs(X_test)

    if model_name == "custom":
        model, vectorizer = build_custom_model(num_classes=len(set(y_train)))
    else:
        model, vectorizer = build_pretrained_style_model(num_classes=len(set(y_train)))

    vectorizer.adapt(X_train["text"].astype(str).values)
    model.fit(train_inputs, y_train, validation_split=0.2, epochs=3, batch_size=32, verbose=1)
    model.evaluate(test_inputs, y_test, verbose=1)
    model.save(MODELS_DIR / f"routing_{model_name}.keras")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["custom", "pretrained"], default="custom")
    args = parser.parse_args()
    main(args.model)
