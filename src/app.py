from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

# Initialize the Flask app
app = Flask(__name__)

# Paths for data and models
RAW_DATA_PATH = "data/raw/manufacturing_defect_dataset.csv"
MODEL_PATH = "models/defect_prediction_model.pkl"

# Upload Endpoint
@app.route('/upload', methods=['POST'])
def upload():
    try:
        # Check if a file is uploaded
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request."}), 400
        
        file = request.files['file']
        # Save the file to 'data/raw'
        os.makedirs("data/raw", exist_ok=True)
        file.save(RAW_DATA_PATH)

        return jsonify({"message": f"File uploaded successfully to {RAW_DATA_PATH}"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Train Endpoint
@app.route('/train', methods=['POST'])
def train():
    try:
        # Check if dataset exists
        if not os.path.exists(RAW_DATA_PATH):
            return jsonify({"error": "Dataset not found. Please upload the dataset first."}), 400

        # Load the dataset
        df = pd.read_csv(RAW_DATA_PATH)

        # Separate features and target variable
        if "DefectStatus" not in df.columns:
            return jsonify({"error": "Target column 'DefectStatus' not found in dataset."}), 400
        
        X = df.drop(columns=["DefectStatus"])
        y = df["DefectStatus"]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Save the model
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        return jsonify({
            "message": "Model trained successfully.",
            "accuracy": accuracy,
            "f1_score": f1
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Predict Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the model exists
        if not os.path.exists(MODEL_PATH):
            return jsonify({"error": "Model not found. Please train the model first."}), 400
        
        # Load the model
        model = joblib.load(MODEL_PATH)

        # Get input JSON
        input_data = request.get_json()

        if not input_data:
            return jsonify({"error": "No input data provided."}), 400

        # Convert JSON to DataFrame
        df = pd.DataFrame([input_data])

        # Validate input columns
        required_columns = model.feature_names_in_
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            return jsonify({"error": f"Missing required input features: {missing_cols}"}), 400

        # Reorder and filter columns to match the model's training data
        df = df[required_columns]

        # Make prediction and get probabilities
        probabilities = model.predict_proba(df)
        predicted_class = int(model.predict(df)[0])  # Predicted class
        confidence = float(probabilities[0][predicted_class])  # Confidence for the predicted class

        # Map predicted class to Downtime label
        class_mapping = {0: "No", 1: "Yes"}
        downtime = class_mapping[predicted_class]

        # Return response with Downtime and Confidence
        return jsonify({
            "Downtime": downtime,
            "Confidence": round(confidence, 2)
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Main Function
if __name__ == "__main__":
    app.run(debug=True, port=5000)
