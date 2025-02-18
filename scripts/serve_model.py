from flask import Flask, request, jsonify
import pickle
import numpy as np
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Load the trained model
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = np.array(data["features"]).reshape(1, -1)
        
        logging.info(f"Received prediction request: {data}")

        prediction = model.predict(features)
        
        logging.info(f"Prediction result: {prediction[0]}")
        
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
