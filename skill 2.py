import os
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Check if dataset exists
if not os.path.exists("student_marks.csv"):
    print("Error: student_marks.csv not found!")
    exit()

# Read dataset
data = pd.read_csv("student_marks.csv")

# Check required columns
if "Hours" not in data.columns or "Marks" not in data.columns:
    print("Error: Dataset must contain 'Hours' and 'Marks' columns.")
    print("Available columns:", list(data.columns))
    exit()

# Features and target
X = data[["Hours"]]
y = data["Marks"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, predictions)

print("Actual Marks:")
print(y_test.values)

print("\nPredicted Marks:")
print(predictions)

print("\nMean Squared Error:", mse)

# Predict for new input
hours = float(input("\nEnter study hours: "))
predicted_marks = model.predict(pd.DataFrame([[hours]], columns=["Hours"]))

print(f"Predicted Marks for {hours} hours = {predicted_marks[0]:.2f}")