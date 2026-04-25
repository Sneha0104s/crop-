from flask import Flask, render_template, request
import joblib
import pandas as pd


app = Flask(__name__)


# ==========================================================
# LOAD MODELS & ENCODERS
# ==========================================================

# Crop Model (Decision Tree)
crop_model = joblib.load("crop_recommendation_dt_model.pkl")

# Fertilizer Model (Decision Tree)
fert_model = joblib.load("fertilizer_dt_model.pkl")

# Encoders
soil_encoder = joblib.load("soil_encoder.pkl")
crop_encoder = joblib.load("crop_encoder.pkl")
fert_encoder = joblib.load("fertilizer_encoder.pkl")

# Feature columns for fertilizer
feature_columns = joblib.load("feature_columns.pkl")

print("✅ Models & Encoders Loaded Successfully")


# ==========================================================
# HOME PAGE
# ==========================================================
@app.route("/")
def home():
    return render_template("home.html")


# ==========================================================
# ABOUT PAGE
# ==========================================================
@app.route("/about")
def about():
    return render_template("about.html")


# ==========================================================
# CROP PAGE
# ==========================================================
@app.route("/crop")
def crop_page():
    return render_template("crop.html")


# ==========================================================
# FERTILIZER PAGE
# ==========================================================
@app.route("/fertilizer")
def fertilizer_page():
    return render_template("fertilizer.html")


# ==========================================================
# CROP PREDICTION
# ==========================================================
@app.route("/predict_crop", methods=["POST"])
def predict_crop():

    try:
        # Get form values
        data = {
            "N": float(request.form["N"]),
            "P": float(request.form["P"]),
            "K": float(request.form["K"]),
            "temperature": float(request.form["temperature"]),
            "humidity": float(request.form["humidity"]),
            "ph": float(request.form["ph"]),
            "rainfall": float(request.form["rainfall"])
        }

        # Create DataFrame
        input_df = pd.DataFrame([data])

        # Predict
        result = crop_model.predict(input_df)[0]

        return render_template(
            "result.html",
            crop_result=result
        )

    except Exception as e:
        return f"Crop Prediction Error: {str(e)}"


# ==========================================================
# FERTILIZER PREDICTION
# ==========================================================
@app.route("/predict_fertilizer", methods=["POST"])
def predict_fertilizer():

    try:
        # Get form values
        data = {
            "Temperature": float(request.form["Temperature"]),
            "Humidity": float(request.form["Humidity"]),
            "Moisture": float(request.form["Moisture"]),
            "Soil Type": request.form["Soil_Type"],
            "Crop Type": request.form["Crop_Type"],
            "Nitrogen": float(request.form["Nitrogen"]),
            "Potassium": float(request.form["Potassium"]),
            "Phosphorous": float(request.form["Phosphorous"])
        }

        # Encode categorical values
        data["Soil Type"] = soil_encoder.transform(
            [data["Soil Type"]]
        )[0]

        data["Crop Type"] = crop_encoder.transform(
            [data["Crop Type"]]
        )[0]

        # Create DataFrame
        input_df = pd.DataFrame([data])

        # Add missing columns (safety)
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder columns
        input_df = input_df[feature_columns]

        # Predict
        fert_encoded = fert_model.predict(input_df)[0]

        fert_name = fert_encoder.inverse_transform(
            [fert_encoded]
        )[0]

        return render_template(
            "fertilizer_result.html",
            fertilizer_result=fert_name
        )

    except Exception as e:
        return f"Fertilizer Prediction Error: {str(e)}"


# ==========================================================
# RUN APP
# ==========================================================
if __name__ == "__main__":
    app.run(debug=True)