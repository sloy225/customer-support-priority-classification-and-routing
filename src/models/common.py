import pandas as pd
from src.utils.config import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS

def df_to_inputs(df: pd.DataFrame):
    inputs = {"text": df["text"].astype(str).values}
    for col in CATEGORICAL_COLUMNS:
        inputs[col] = df[col].astype(str).values
    for col in NUMERICAL_COLUMNS:
        inputs[col] = df[col].astype("float32").values
    return inputs
