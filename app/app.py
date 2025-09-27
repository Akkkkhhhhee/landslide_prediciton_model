from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load your saved model and feature columns
model = joblib.load('model/landslide_model.pkl')
feature_columns = joblib.load('model/feature_columns.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Parse inputs from the form
    rain = float(request.form['rainfall'])
    slope = float(request.form['slope_angle'])
    soil_saturation = float(request.form['soil_saturation'])
    vegetation = float(request.form['vegetation_cover'])
    earthquake = float(request.form['earthquake_activity'])
    proximity = float(request.form['proximity_to_water'])
    soil_type = request.form['soil_type']

    # Prepare input dictionary for prediction
    input_dict = {col: 0 for col in feature_columns}
    input_dict['Rainfall_mm'] = rain
    input_dict['Slope_Angle'] = slope
    input_dict['Soil_Saturation'] = soil_saturation
    input_dict['Vegetation_Cover'] = vegetation
    input_dict['Earthquake_Activity'] = earthquake
    input_dict['Proximity_to_Water'] = proximity

    # Set the soil type one-hot encoded column
    soil_col = f"Soil_Type_{soil_type}"
    if soil_col in input_dict:
        input_dict[soil_col] = 1

    input_df = pd.DataFrame([input_dict])

    # Make prediction
    prediction = model.predict(input_df)[0]
    result = "Landslide" if prediction == 1 else "No Landslide"

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
