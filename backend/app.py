from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "soil_health_model.pkl")
model = joblib.load(MODEL_PATH)

# ✅ Add the missing two features: ph and nitrogen
class SoilData(BaseModel):
    ndvi: float
    moisture: float
    vegetation_cover: float
    ph: float
    nitrogen: float

@app.post("/predict")
def predict_soil_health(data: SoilData):
    # Order of features must match training order
    x = np.array([[data.ndvi, data.moisture, data.vegetation_cover, data.ph, data.nitrogen]])
    score = model.predict(x)[0]
    label = "Healthy" if score > 70 else "Moderate" if score > 40 else "Degraded"
    return {"score": round(float(score), 2), "status": label}
