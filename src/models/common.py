import pandas as pd
import tensorflow as tf
from src.utils.config import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS


def df_to_inputs(df: pd.DataFrame):
    inputs = {
        "text": tf.constant(
            df["text"].astype(str).values,
            dtype=tf.string
        )
    }

    for col in CATEGORICAL_COLUMNS:
        inputs[col] = tf.constant(
            df[col].astype(str).values,
            dtype=tf.string
        )

    for col in NUMERICAL_COLUMNS:
        inputs[col] = tf.constant(
            df[col].astype("float32").values,
            dtype=tf.float32
        )

    return inputs