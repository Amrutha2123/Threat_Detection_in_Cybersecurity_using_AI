import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset
file_path = '/home/amrutha-m-y/Desktop/ThreatDetectionProject/Data/Train_data.csv'
df = pd.read_csv(file_path)

# Print column names to verify the target column
print("Dataset Columns:", df.columns)

# Define the target column
TARGET_COLUMN = 'class'

# Ensure the target column exists
if TARGET_COLUMN not in df.columns:
    raise KeyError(f"Error: The target column '{TARGET_COLUMN}' does not exist in the dataset.")

# Separate features and target
X = df.drop(columns=[TARGET_COLUMN])  # Features
y = df[TARGET_COLUMN]  # Target variable

# Identify categorical columns (non-numeric)
categorical_columns = X.select_dtypes(include=['object']).columns
print("Categorical Columns:", categorical_columns)

# Convert categorical columns to numeric using Label Encoding
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])  # Convert string labels to numbers
    label_encoders[col] = le

# Handle missing values and infinity issues
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(0, inplace=True)  # Replace NaN values

# Normalize the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Save the trained model
joblib.dump(model, '/home/amrutha-m-y/Desktop/ThreatDetectionProject/Backend/model.pkl')

# Save label encoders for future use (optional)
joblib.dump(label_encoders, '/home/amrutha-m-y/Desktop/ThreatDetectionProject/Backend/label_encoders.pkl')

print("Model training completed successfully and saved as 'model.pkl'.")

