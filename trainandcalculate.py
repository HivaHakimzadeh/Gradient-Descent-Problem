import numpy as np
import pandas as pd
import logging as lg
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

pd.set_option('display.max_columns', None)

# Import the CSV file into a DataFrame
df = pd.read_csv("https://github.com/77EminSarac77/Dataset-for-Linear-Regression/blob/main/Diabetespred.csv?raw=true")

# Define features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Pre-processing data
# There are no missing values also all the feature variables are numerical so there is no need
# to convert

# Splitting up the dataset into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# # Adding column of 1's to X_train and X_test
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

# # Parameters required for Gradient Descent
alpha = 0.00001  # learning rate
m = X_train.shape[0]  # no. of samples
np.random.seed(10)
theta = np.random.rand(X_train.shape[1])  # the weights
tolerance = 1e-9
max_iters = 1000000

# Convert Y_train to a NumPy array
Y_train = np.array(Y_train)


def gradient_descent(X, y, m, theta, alpha):
    cost_list = []  # to record all cost values
    theta_list = []  # to record all theta values
    prediction_list = []
    run = True
    i = 0
    while run and i < max_iters:
        prediction = np.dot(X, theta)  # predicted y values
        prediction_list.append(prediction)
        error = prediction - y
        cost = (1 / (2 * m)) * np.dot(error.T, error)  # cost function
        if i > 0 and abs(cost_list[-1] - cost) < tolerance:
            break
        cost_list.append(cost)
        theta = theta - (alpha * (1 / m) * np.dot(X.T, error))  # update weights
        theta_list.append(theta)
        i += 1
    return prediction_list, cost_list, theta_list


# Train the model using gradient descent
prediction_list, cost_list, theta_list = gradient_descent(X_train, Y_train, m, theta, alpha)
theta = theta_list[-1]

# Make predictions on the test set
predictions_test = np.dot(X_test, theta)

# Calculate Mean Squared Error (MSE) on the test set
mse = ((predictions_test - Y_test) ** 2).mean()
print('Mean Squared Error from Gradient Descent prediction : {}'.format(round(mse, 3)))

# Testing part
# Mean Absolute Error (MAE)
mae = mean_absolute_error(Y_test, predictions_test)

# R-squared (R2)
r2 = r2_score(Y_test, predictions_test)

print('Mean Absolute Error from Gradient Descent prediction: {}'.format(round(mae, 3)))
print('R-squared from Gradient Descent prediction: {}'.format(round(r2, 3)))


# Generating log file (the code for this will be commented so that it doesn't add anything to the log file that
# I turned in while the TA runs my code)
lg.basicConfig(filename="part1.log", level=lg.INFO)

lg.info('\n' + 'tolerance: ' + str(tolerance) + '\n' + 'max iterations: ' + str(max_iters) + '\n' + 'learning rate: ' +
        str(alpha) + '\n' + 'MSE: ' + str(mse) + '\n')
