import tensorflow as tf
from tensorflow import keras
from src.utils.config import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS

def build_custom_model(num_classes: int, vocab_size: int = 20000, sequence_length: int = 200):
    text_input = keras.Input(shape=(1,), dtype=tf.string, name="text")
    vectorizer = keras.layers.TextVectorization(max_tokens=vocab_size, output_mode="int", output_sequence_length=sequence_length)
    x_text = vectorizer(text_input)
    x_text = keras.layers.Embedding(vocab_size, 128)(x_text)
    x_text = keras.layers.Bidirectional(keras.layers.LSTM(64))(x_text)

    inputs = {"text": text_input}
    features = [x_text]

    for col in CATEGORICAL_COLUMNS:
        inp = keras.Input(shape=(1,), dtype=tf.string, name=col)
        lookup = keras.layers.StringLookup(output_mode="one_hot")
        encoded = lookup(inp)
        inputs[col] = inp
        features.append(encoded)

    for col in NUMERICAL_COLUMNS:
        inp = keras.Input(shape=(1,), dtype=tf.float32, name=col)
        norm = keras.layers.Normalization()
        encoded = norm(inp)
        inputs[col] = inp
        features.append(encoded)

    x = keras.layers.concatenate(features)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    output = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model, vectorizer
