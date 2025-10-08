import joblib
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "soil_health_model.pkl")

# Load the trained model
model = joblib.load(MODEL_PATH)

def predict_soil_health(ndvi: float, moisture: float, vegetation_cover: float):
    """Predict soil health score and label"""
    x = np.array([[ndvi, moisture, vegetation_cover]])
    score = model.predict(x)[0]
    label = "Healthy" if score > 70 else "Moderate" if score > 40 else "Degraded"
    return {"score": round(float(score), 2), "status": label}
