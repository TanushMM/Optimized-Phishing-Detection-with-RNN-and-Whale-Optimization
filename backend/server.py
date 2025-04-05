import nest_asyncio
import pickle
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import uvicorn

nest_asyncio.apply()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained RNN model
model = load_model("./model.h5")

# Load trained MinMaxScaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

class InputFeatures(BaseModel):
    features: list[float]  # Input must be exactly 22 scaled features

@app.get("/")
def read_root():
    return {"message": "Phishing Detector API is live!"}

@app.post("/predict")
def predict(input: InputFeatures):
    try:
        # Convert to NumPy array
        input_array = np.array([input.features])  # Shape: (1, 22)

        # Scale the input using the loaded scaler
        scaled = scaler.transform(input_array)

        # Reshape for RNN: (batch_size, time_steps, features)
        reshaped = scaled.reshape((1, scaled.shape[1], 1))

        # Run model prediction
        prediction = model.predict(reshaped)

        print(prediction)

        # Generate label based on prediction threshold
        label = "phishing" if prediction[0][0] >= 0.5 else "safe"

        return {"prediction": label}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)