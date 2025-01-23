Predictive Analysis for Manufacturing Operations

Objective:- This project involves creating a RESTful API that performs predictive analysis on manufacturing data to predict machine downtime or production defects. The API provides endpoints for uploading data, training a model, and making predictions.

Setup Instructions:-
Prerequisites
Python: Ensure Python 3.8+ is installed on your system.
Libraries: Install the necessary Python libraries using the provided requirements.txt file.

Installation Steps
Clone the repository:
git clone <https://github.com/Varun-gabhane/PredictiveAnalysis-Manufacturing>
cd <repository-folder>

Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate # For Linux/Mac
venv\Scripts\activate   # For Windows

Install dependencies:
pip install -r requirements.txt

Run the Flask app:
python src/app.py

API Endpoints

1. Upload Endpoint
URL: /upload
Method: POST
Description: Accepts a CSV file containing manufacturing data and saves it for training.
Request:
Form-data:
Key: file
Value: CSV file (e.g., manufacturing_defect_dataset.csv)
Response:
{
  "message": "File uploaded successfully to [data/raw/manufacturing_defect_dataset.csv]"
}

2. Train Endpoint
URL: /train
Method: POST
Description: Trains the predictive model using the uploaded dataset and returns performance metrics.
Request: No input required.
Response: {
  "accuracy": 0.95,
  "f1_score": 0.93
}

3. Predict Endpoint
URL: /predict
Method: POST
Description: Accepts JSON input with manufacturing parameters and predicts machine downtime.
Request (Example):
{
  "Temperature": 80,
  "Run_Time": 120
}
Response (Example):
{
  "Downtime": "Yes",
  "Confidence": 0.85
}

Example API Requests and Responses
Example 1: Upload Dataset
Request:
POST /upload
Form-Data:
- Key: file
- Value: manufacturing_defect_dataset.csv
Response:
{
  "message": "File uploaded successfully to [data/raw/manufacturing_defect_dataset.csv]"
}

Example 2: Train Model
Request:
POST /train
Response:
{
  "accuracy": 0.95,
  "f1_score": 0.93
}

Example 3: Predict Downtime
Request:
POST /predict
Body (JSON):
{
  "Temperature": 80,
  "Run_Time": 120
}
Response:
{
  "Downtime": "Yes",
  "Confidence": 0.85
}

Folder Structure
data/
raw/: Contains raw uploaded data.
processed/: Contains cleaned or preprocessed data.

models/
Stores trained model files (e.g., defect_prediction_model.pkl).

notebooks/
Jupyter notebooks for exploratory data analysis (optional).

src/
app.py: Flask application script.
preprocess.py: Data preprocessing logic.
training_model.py: Model training and evaluation logic.

requirements.txt: Python dependencies.

Additional Notes
Ensure that the dataset has relevant columns for training, such as Machine_ID, Temperature, Run_Time, and Downtime_Flag.
The model currently uses a Decision Tree Classifier, which can be swapped for a different supervised learning algorithm if needed.

Troubleshooting
Error: "Dataset not found": Upload the dataset using the /upload endpoint.
Environment issues: Ensure dependencies in requirements.txt are installed.
