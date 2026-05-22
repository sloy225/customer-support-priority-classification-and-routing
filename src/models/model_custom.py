import tensorflow as tf
from tensorflow import keras
from src.utils.config import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS


def build_custom_model(
    num_classes: int,
    categorical_vocabularies: dict,
    numerical_means: dict,
    numerical_vars: dict,
    text_corpus: list,          # corpus d'entraînement pour adapter le vectorizer
    vocab_size: int = 20000,
    sequence_length: int = 200
):
    # ── Branche texte ──────────────────────────────────────────────────────
    text_input = keras.Input(shape=(1,), dtype=tf.string, name="text")

    text_vectorizer = keras.layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=sequence_length
    )
    # Adaptation sur le corpus réel AVANT la construction du graphe
    text_vectorizer.adapt(text_corpus)

    x_text = text_vectorizer(text_input)
    x_text = keras.layers.Embedding(input_dim=vocab_size, output_dim=128, mask_zero=True)(x_text)
    x_text = keras.layers.Bidirectional(keras.layers.LSTM(64, dropout=0.3))(x_text)
    x_text = keras.layers.Dropout(0.4)(x_text)

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

    return model, text_vectorizer