import pandas as pd
import math
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Read dataset into a DataFrame
df = pd.read_csv("E:\Foundation of Data Science\po2_data.csv")

# Defining features (X) and target (y)
X = df.drop(columns=['subject#', 'age', 'sex', 'test_time', 'motor_updrs', 'total_updrs'])
y = df['total_updrs']

# List of test sizes for 50-50, 60-40, 70-30, 80-20 splits
splits = [0.5, 0.4, 0.3, 0.2]

for test_size in splits:
    print(f'\nPerformance Metrics For Total_UPDRS with {int((1-test_size)*100)}-{int(test_size*100)} train-test split:')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Compute metrics for linear regression model
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    r2 = metrics.r2_score(y_test, y_pred)
    adjusted_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
    nrmse = rmse / (y_test.max() - y_test.min())

    print("Linear Regression Metrics:")
    print("MAE: ", mae)
    print("MSE: ", mse)
    print("RMSE: ", rmse)
    print("NRMSE: ", nrmse)  
    print("R^2: ", r2)
    print("Adjusted R^2: ", adjusted_r2)
