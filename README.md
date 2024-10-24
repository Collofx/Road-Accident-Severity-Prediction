# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

# Load your dataset (replace with your actual dataset file)
data = pd.read_csv('accident_data.csv')

# Assume these are your columns
# Independent variables: 'Weather', 'RoadCondition', 'TimeOfDay', 'Speed', 'DriverBehavior', 'VehicleType', 'RoadType'
# Dependent variable: 'Severity'
X = data[['Weather', 'RoadCondition', 'TimeOfDay', 'Speed', 'DriverBehavior', 'VehicleType', 'RoadType']]
y = data['Severity']

# Convert categorical data into dummy variables if needed
X = pd.get_dummies(X, drop_first=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model for future use
joblib.dump(model, 'accident_severity_model.pkl')

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Example of using the model to predict accident severity for hypothetical data
example_data = np.array([[1, 0, 0, 60, 1, 0, 1]])  # Replace with actual feature data
predicted_severity = model.predict(example_data)
print(f"Predicted Accident Severity: {predicted_severity}")
