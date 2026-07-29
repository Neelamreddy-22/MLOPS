import os
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Display current working directory
print("Current Working Directory:", os.getcwd())

# CSV file name
file_name = "student_marks.csv"

# Check if file exists
if not os.path.exists(file_name):
    print(f"Error: '{file_name}' not found.")
    print("Place the CSV file in the same folder as this Python file.")
    exit()

# Read dataset
data = pd.read_csv(file_name)

# Display first few rows
print("\nDataset:")
print(data.head())

# Features and target
X = data[["Hours"]]
y = data["Marks"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Decision Tree Regressor
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, predictions)

print("\nPredicted Marks:", predictions)
print("Actual Marks:", list(y_test))
print("Mean Squared Error:", mse)

# Predict for new data
hours = float(input("\nEnter study hours to predict marks: "))
predicted_marks = model.predict([[hours]])

print(f"Predicted Marks for {hours} hours = {predicted_marks[0]:.2f}")