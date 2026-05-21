import re

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def combine_texts(*values) -> str:
    parts = [str(v).strip() for v in values if str(v) != 'nan']
    return " ".join(parts)

def map_routing_label(ticket_type: str) -> str:
    mapping = {
        "Refund request": "retour_produit",
        "Billing inquiry": "facturation",
        "Technical issue": "support_technique",
        "Cancellation request": "annulation",
        "Product inquiry": "information_produit",
    }
    return mapping.get(str(ticket_type).strip(), "autre")
