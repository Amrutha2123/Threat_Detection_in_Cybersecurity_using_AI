import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

# Load the preprocessed test data
test_data = pd.read_csv("path_to_processed_test_data.csv")  # Replace with the actual path

# Load the trained model
model = joblib.load('trained_model.pkl')  # Load the saved model

# Features and target
X_test = test_data.drop('class', axis=1)
y_test = test_data['class']

# Test the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")
