#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load our trained AI model
model = joblib.load("modele_rf_serre.joblib")

app = FastAPI(title="Greenhouse Irrigation AI")


# Sensor data received from ESP32
class SensorData(BaseModel):
    heure: int
    mois: int
    temp_air: float
    hum_air: float
    hum_sol: float
    luminosite_lux: float
    pluie: int


# Test endpoint
@app.get("/")
def home():
    return {"message": "Greenhouse AI is running!"}


# AI prediction endpoint
@app.post("/predict")
def predict(data: SensorData):

    inputs = pd.DataFrame([{
        "heure": data.heure,
        "mois": data.mois,
        "temp_air": data.temp_air,
        "hum_air": data.hum_air,
        "hum_sol": data.hum_sol,
        "luminosite_lux": data.luminosite_lux,
        "pluie": data.pluie
    }])

    prediction = model.predict(inputs)[0]

    return {
        "volume_eau_L_m2": round(float(prediction), 3)
    }


# In[ ]:





# In[ ]:




