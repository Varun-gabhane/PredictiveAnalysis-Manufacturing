import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Training Function
def train_model(input_file, model_output_path):
    # Load preprocessed data
    df = pd.read_csv('data/raw/manufacturing_defect_dataset.csv')

    # Split into features (X) and target (y)
    X = df.drop(columns=['DefectStatus'])  # Features
    y = df['DefectStatus']               # Target variable

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Model Training Complete.")
    print(f"Accuracy on Test Set: {accuracy}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save the trained model
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"Model saved to {model_output_path}")

# Paths
processed_data_path = "data/processed/cleaned_data.csv"
model_path = "models/defect_prediction_model.pkl"

# Run training
if __name__ == "__main__":
    train_model(processed_data_path, model_path)
