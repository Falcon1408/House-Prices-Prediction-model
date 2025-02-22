import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

def gradient_descent(df, learning_rate, iterations):
    data = df.to_numpy()
    X = data[:, :-1]  # Features
    y = data[:, -1]   # Target
    rows, num_features = X.shape

    # Initialize parameters and intercept
    parameters = np.zeros(num_features)
    intercept = 0

    # Perform gradient descent
    for i in range(iterations):
        predictions = np.dot(X, parameters) + intercept
        errors = y - predictions

        del_parameters = -(2 / rows) * np.dot(X.T, errors)  # Gradient for parameters
        del_intercept = -(2 / rows) * np.sum(errors)        # Gradient for intercept

        parameters -= learning_rate * del_parameters
        intercept -= learning_rate * del_intercept

    return parameters, intercept

def evaluate_algorithm(file_name, learning_rate=0.0001, iterations=1000):
    df = pd.read_csv(file_name)
    parameters, intercept = gradient_descent(df, learning_rate, iterations)
    print("Final Parameters (Slopes):", parameters)
    print("Final Intercept:", intercept)
    return parameters, intercept

def predict(test_file, parameters, intercept):
    test_df = pd.read_csv(test_file)
    test_data = test_df.to_numpy()
    X_test = test_data[:, :-1]
    y_test = test_data[:, -1]  # Actual target values
    predictions = np.dot(X_test, parameters) + intercept
    return predictions, y_test

def calculate_accuracy(predictions, y_test):
    # Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, predictions)
    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    # R-squared (R²)
    r2 = r2_score(y_test, predictions)
    
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("R-squared (R²):", r2)

def main():
    train_file = "train_data.csv"
    test_file = "test_data.csv"
    learning_rate = 0.001
    iterations = 1000

    # Train the model
    parameters, intercept = evaluate_algorithm(train_file, learning_rate, iterations)

    # Make predictions on the test data
    predictions, y_test = predict(test_file, parameters, intercept)

    # Output the predictions
    print("Predictions for Test Data:", predictions)

    # Calculate the accuracy of the model
    calculate_accuracy(predictions, y_test)

if __name__ == "__main__":
    main()

