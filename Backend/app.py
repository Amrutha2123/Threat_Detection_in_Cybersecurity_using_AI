from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# ✅ Load trained model correctly
model_path = "/home/amrutha-m-y/Desktop/ThreatDetectionProject/Backend/model.pkl"
try:
    model = joblib.load(model_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

@app.route('/', methods=['GET'])
def home():
    return "Threat Detection API is running! Use /predict to make predictions."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request (JSON format)
        data = request.get_json()
        print(f"Data received: {data}")  # Print received data to check if it's correct
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        print(f"Data converted to DataFrame:\n{df}")  # Print DataFrame for debugging
        
        # Predict using the trained model
        prediction = model.predict(df)
        print(f"Prediction: {prediction}")  # Print the prediction result
        
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Change port if needed

