import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_and_preprocess(file_path):
    try:
        # Step 1: Load the dataset
        data = pd.read_csv(file_path)
        print(f"Processing file: {file_path}")
        
        # Step 2: Check initial shape
        print(f"Number of rows and columns before preprocessing: {data.shape}")

        # Step 3: Fill missing values (example: fill with 0, or forward fill)
        data_filled = data.fillna(0)  # You can change this strategy

        # Step 4: Check for any missing values after filling
        if data_filled.isna().sum().sum() > 0:
            print(f"Warning: There are still missing values after filling.")
        
        print(f"Number of rows after filling missing values: {data_filled.shape[0]}")

        # Step 5: Drop rows that still contain NaN or invalid values (optional)
        # data_filled = data_filled.dropna()  # Uncomment if you want to drop rows

        # Step 6: Select only numeric columns for scaling
        numeric_data = data_filled.select_dtypes(include=['float64', 'int64'])
        print(f"Number of rows and columns after selecting numeric data: {numeric_data.shape}")

        if numeric_data.empty:
            raise ValueError("Dataset contains no numeric data after preprocessing.")

        # Step 7: Check for infinity or very large values
        if np.isinf(numeric_data.values).any():
            print("Warning: Dataset contains infinity values. Replacing them with NaN.")
            numeric_data.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Step 8: Replace NaN values created by infinity replacement (optional: use mean, median, etc.)
        numeric_data = numeric_data.fillna(0)  # Or use .fillna(numeric_data.mean())

        # Step 9: Check for extremely large values (optional: clip values to a reasonable range)
        numeric_data = numeric_data.clip(-1e10, 1e10)  # Clip extreme values to a specified range (change as needed)

        # Step 10: Apply StandardScaler
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        print(f"Scaled data shape: {scaled_data.shape}")

        return scaled_data

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None


# Example usage with Friday and Monday data
friday_scaled = load_and_preprocess('/home/amrutha-m-y/Desktop/ThreatDetectionProject/Data/Friday.csv')
monday_scaled = load_and_preprocess('/home/amrutha-m-y/Desktop/ThreatDetectionProject/Data/Monday.csv')

if friday_scaled is not None:
    print("Friday data preprocessing successful.")

if monday_scaled is not None:
    print("Monday data preprocessing successful.")

