from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "customer_support_tickets.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "tickets_ready.csv"
MODELS_DIR = BASE_DIR / "models"
TEXT_COLUMNS = ["Ticket Subject", "Ticket Description"]
CATEGORICAL_COLUMNS = ["Customer Gender", "Product Purchased", "Ticket Type", "Ticket Channel"]
NUMERICAL_COLUMNS = ["Customer Age"]
PRIORITY_TARGET = "priority_label"
ROUTING_TARGET = "routing_label"
RANDOM_STATE = 42
