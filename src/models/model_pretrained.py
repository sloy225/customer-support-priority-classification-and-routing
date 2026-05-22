import tensorflow as tf
from tensorflow import keras
from src.utils.config import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS


def build_pretrained_style_model(
    num_classes: int,
    categorical_vocabularies: dict,
    numerical_means: dict,
    numerical_vars: dict,
    text_corpus: list,
    vocab_size: int = 20000
):
    # ── Branche texte TF-IDF ───────────────────────────────────────────────
    text_input = keras.Input(shape=(1,), dtype=tf.string, name="text")

    vectorizer = keras.layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="tf_idf"
    )
    vectorizer.adapt(text_corpus)

    x_text = vectorizer(text_input)
    x_text = keras.layers.Flatten()(x_text)          # ← corrige le shape (None,1,N) → (None,N)
    x_text = keras.layers.Dense(256, activation="relu")(x_text)
    x_text = keras.layers.BatchNormalization()(x_text)
    x_text = keras.layers.Dropout(0.4)(x_text)
    x_text = keras.layers.Dense(128, activation="relu")(x_text)
    x_text = keras.layers.Dropout(0.3)(x_text)

    inputs = {"text": text_input}
    features = [x_text]

    # ── Branches catégorielles ─────────────────────────────────────────────
    for col in CATEGORICAL_COLUMNS:
        inp = keras.Input(shape=(1,), dtype=tf.string, name=col)
        lookup = keras.layers.StringLookup(
            vocabulary=categorical_vocabularies[col],
            output_mode="one_hot"
        )
        encoded = keras.layers.Flatten()(lookup(inp))
        inputs[col] = inp
        features.append(encoded)

    # ── Branche numérique ──────────────────────────────────────────────────
    for col in NUMERICAL_COLUMNS:
        inp = keras.Input(shape=(1,), dtype=tf.float32, name=col)
        norm = keras.layers.Normalization(
            axis=-1,
            mean=[numerical_means[col]],
            variance=[numerical_vars[col]]
        )
        inputs[col] = inp
        features.append(norm(inp))

    # ── Fusion ─────────────────────────────────────────────────────────────
    x = keras.layers.Concatenate()(features)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    output = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model, vectorizer