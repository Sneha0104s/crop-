# =========================
# 1. Import Libraries
# =========================
import joblib
import numpy as np
import pandas as pd


# =========================
# 2. Load Trained Model
# =========================
model = joblib.load("crop_recommendation_dt_model.pkl")

print("✅ Model Loaded Successfully")


# =========================
# 3. User Input Function
# =========================
def get_user_input():

    print("\nEnter Soil and Climate Details:\n")

    N = float(input("Enter Nitrogen (N): "))
    P = float(input("Enter Phosphorus (P): "))
    K = float(input("Enter Potassium (K): "))

    temperature = float(input("Enter Temperature (°C): "))
    humidity = float(input("Enter Humidity (%): "))
    ph = float(input("Enter pH value: "))
    rainfall = float(input("Enter Rainfall (mm): "))

    data = {
        "N": [N],
        "P": [P],
        "K": [K],
        "temperature": [temperature],
        "humidity": [humidity],
        "ph": [ph],
        "rainfall": [rainfall]
    }

    return pd.DataFrame(data)


# =========================
# 4. Take Input
# =========================
input_data = get_user_input()


# =========================
# 5. Predict Crop
# =========================
prediction = model.predict(input_data)


# =========================
# 6. Show Result
# =========================
print("\n🌱 Recommended Crop is:", prediction[0])