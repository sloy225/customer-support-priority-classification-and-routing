import pandas as pd
from src.utils.config import RAW_DATA_PATH

def load_raw_data(path=RAW_DATA_PATH):
    return pd.read_csv(path)
