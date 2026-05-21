from fastapi import FastAPI
from pydantic import BaseModel
from src.models.predict import predict_ticket

app = FastAPI(title="Ticket Prioritization and Routing API")

class TicketPayload(BaseModel):
    text: str
    Customer_Gender: str
    Product_Purchased: str
    Ticket_Type: str
    Ticket_Channel: str
    Customer_Age: float

@app.get("/")
def root():
    return {"message": "API de priorisation et de routage intelligent de tickets"}

@app.post("/predict")
def predict(payload: TicketPayload):
    sample = {
        "text": payload.text,
        "Customer Gender": payload.Customer_Gender,
        "Product Purchased": payload.Product_Purchased,
        "Ticket Type": payload.Ticket_Type,
        "Ticket Channel": payload.Ticket_Channel,
        "Customer Age": payload.Customer_Age,
    }
    return {
        "predicted_priority": predict_ticket(sample, task="priority", model_name="custom"),
        "predicted_routing": predict_ticket(sample, task="routing", model_name="custom")
    }
