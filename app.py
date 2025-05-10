from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import json

app = Flask(__name__)
CORS(app)

# Load model and feature list
model = joblib.load("model/heart_disease_risk_model.pkl")
with open("model/features.json", "r") as f:
    FEATURES = json.load(f)

@app.route('/')
def index():
    return "Heart Disease Risk Prediction API is running."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_values = [data[feature] for feature in FEATURES]
        input_array = np.array([input_values])

        prediction = model.predict(input_array)[0]
        probability = model.predict_proba(input_array)[0][1]
        label = "At Risk" if prediction == 1 else "Low Risk"

        return jsonify({
            'prediction': int(prediction),
            'risk_label': label,
            'risk_probability': round(float(probability), 2)
        })

    except KeyError as e:
        return jsonify({'error': f'Missing input field: {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


