import pandas as pd
from src.data.load_data import load_raw_data
from src.utils.config import PROCESSED_DATA_PATH, TEXT_COLUMNS, CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS
from src.utils.helpers import clean_text, combine_texts, map_routing_label

def prepare_dataset():
    df = load_raw_data().copy()
    keep_cols = TEXT_COLUMNS + CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS + ["Ticket Priority"]
    df = df[keep_cols].dropna(subset=["Ticket Priority"]).copy()
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].fillna("unknown").astype(str)
    for col in NUMERICAL_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())
    df["text"] = df.apply(lambda r: combine_texts(r["Ticket Subject"], r["Ticket Description"]), axis=1)
    df["text"] = df["text"].apply(clean_text)
    df["priority_label"] = df["Ticket Priority"].astype(str)
    df["routing_label"] = df["Ticket Type"].apply(map_routing_label)
    df = df[["text"] + CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS + ["priority_label", "routing_label"]]
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Saved processed dataset to {PROCESSED_DATA_PATH}")
    return df

if __name__ == "__main__":
    prepare_dataset()
