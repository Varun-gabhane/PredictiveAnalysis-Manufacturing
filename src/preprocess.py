import pandas as pd
import os

# Preprocessing Function
def preprocess_data(input_file, output_file):
    # Load raw data
    df = pd.read_csv('data/raw/manufacturing_defect_dataset.csv')
    print("Initial Data Shape:", df.shape)

    # Handle missing values (example: drop rows with NaNs)
    df.dropna(inplace=True)
    print("After Dropping Missing Values:", df.shape)

    # Normalize numerical columns (example: ProductionVolume)
    numerical_cols = ['ProductionVolume', 'ProductionCost']
    for col in numerical_cols:
        df[col] = (df[col] - df[col].mean()) / df[col].std()

    # Save the processed data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

# Paths
raw_data_path = "data/raw/manufacturing_defect_dataset.csv"
processed_data_path = "data/processed/cleaned_data.csv"

# Run preprocessing
if __name__ == "__main__":
    preprocess_data(raw_data_path, processed_data_path)
