import pandas as pd
import math
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Read dataset into a DataFrame
df = pd.read_csv("E:\Foundation of Data Science\po2_data.csv")

# Defining features (X) and target (y)
print('Performance Metrics For Total_UPDRS')
X = df.drop(columns=['subject#', 'age', 'sex', 'test_time', 'motor_updrs', 'total_updrs'])
y = df['total_updrs']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Compute metrics for linear regression model
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
r2 = metrics.r2_score(y_test, y_pred)
adjusted_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
nrmse = rmse / (y_test.max() - y_test.min())  # Adding NRMSE

print("Linear Regression Metrics:")
print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("NRMSE: ", nrmse)  # Printing NRMSE
print("R^2: ", r2)
print("Adjusted R^2: ", adjusted_r2)

# Baseline model
y_base = np.mean(y_train)
y_pred_base = [y_base] * len(y_test)

base_mae = metrics.mean_absolute_error(y_test, y_pred_base)
base_mse = metrics.mean_squared_error(y_test, y_pred_base)
base_rmse = math.sqrt(base_mse)
base_r2 = metrics.r2_score(y_test, y_pred_base)
base_adjusted_r2 = 1 - (1 - base_r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
base_nrmse = base_rmse / (y_test.max() - y_test.min())  # Adding NRMSE for baseline

print("\nBaseline Model Metrics:")
print("Baseline MAE: ", base_mae)
print("Baseline MSE: ", base_mse)
print("Baseline RMSE: ", base_rmse)
print("Baseline NRMSE: ", base_nrmse)  # Printing NRMSE for baseline
print("Baseline R^2: ", base_r2)
print("Baseline Adjusted R^2: ", base_adjusted_r2)
