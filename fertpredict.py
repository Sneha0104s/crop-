# =========================
# 1. Import Libraries
# =========================
import joblib
import pandas as pd
import numpy as np


# =========================
# 2. Load Saved Model & Encoders
# =========================
model = joblib.load("fertilizer_dt_model.pkl")

soil_encoder = joblib.load("soil_encoder.pkl")
crop_encoder = joblib.load("crop_encoder.pkl")
fert_encoder = joblib.load("fertilizer_encoder.pkl")

feature_columns = joblib.load("feature_columns.pkl")

print("✅ Model and Encoders Loaded Successfully")


# =========================
# 3. Take User Input
# =========================
def get_user_input():

    print("\nEnter Soil and Crop Details:\n")

    temperature = float(input("Enter Temperature: "))
    humidity = float(input("Enter Humidity: "))
    moisture = float(input("Enter Moisture: "))

    soil_type = input("Enter Soil Type: ").strip()
    crop_type = input("Enter Crop Type: ").strip()

    nitrogen = float(input("Enter Nitrogen: "))
    potassium = float(input("Enter Potassium: "))
    phosphorous = float(input("Enter Phosphorous: "))

    # Encode categorical values
    soil_encoded = soil_encoder.transform([soil_type])[0]
    crop_encoded = crop_encoder.transform([crop_type])[0]

    # Create DataFrame
    input_data = pd.DataFrame([[
        temperature,
        humidity,
        moisture,
        soil_encoded,
        crop_encoded,
        nitrogen,
        potassium,
        phosphorous
    ]], columns=feature_columns)

    return input_data


# =========================
# 4. Get Input
# =========================
input_df = get_user_input()


# =========================
# 5. Predict Fertilizer
# =========================
prediction = model.predict(input_df)

fertilizer_name = fert_encoder.inverse_transform(prediction)[0]


# =========================
# 6. Show Result
# =========================
print("\n🌱 Recommended Fertilizer:", fertilizer_name)