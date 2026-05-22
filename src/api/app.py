from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from src.models.predict import predict_ticket

app = FastAPI(
    title="Ticket Prioritization & Routing API",
    description="API de priorisation et routage intelligent de tickets support",
    version="1.0.0"
)

# CORS — nécessaire pour un frontend React/Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TicketPayload(BaseModel):
    text: str = Field(..., example="My laptop screen is broken")
    Customer_Gender: str = Field(..., example="Male")
    Product_Purchased: str = Field(..., example="Laptop")
    Ticket_Type: str = Field(..., example="Technical support")
    Ticket_Channel: str = Field(..., example="Email")
    Customer_Age: float = Field(..., example=35.0)
    model_name: str = Field(default="custom", example="custom")


class PredictionResponse(BaseModel):
    predicted_priority: str
    predicted_routing: str
    model_used: str


@app.get("/")
def root():
    return {"message": "API de priorisation et routage intelligent de tickets"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: TicketPayload):
    sample = {
        "text": payload.text,
        "Customer Gender": payload.Customer_Gender,
        "Product Purchased": payload.Product_Purchased,
        "Ticket Type": payload.Ticket_Type,
        "Ticket Channel": payload.Ticket_Channel,
        "Customer Age": payload.Customer_Age,
    }
    try:
        priority = predict_ticket(sample, task="priority", model_name=payload.model_name)
        routing  = predict_ticket(sample, task="routing",  model_name=payload.model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PredictionResponse(
        predicted_priority=priority,
        predicted_routing=routing,
        model_used=payload.model_name
    )