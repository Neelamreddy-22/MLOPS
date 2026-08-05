import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

# Sample dataset
data = pd.DataFrame({
    "Hours": [1, 2, 3, 4, 5],
    "Scores": [10, 20, 30, 40, 50]
})

# Features and Target
X = data[["Hours"]]
y = data["Scores"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Start MLflow tracking
with mlflow.start_run():

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, predictions)

    # Log parameters
    mlflow.log_param("model", "LinearRegression")

    # Log metrics
    mlflow.log_metric("mse", mse)

    # Log the trained model
    mlflow.sklearn.log_model(model, "linear_regression_model")

print("Model trained successfully!")