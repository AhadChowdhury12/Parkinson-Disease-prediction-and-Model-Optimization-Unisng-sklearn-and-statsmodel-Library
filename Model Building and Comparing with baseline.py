import pandas as pd
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Read dataset into a DataFrame
df = pd.read_csv("E:\Foundation of Data Science\po2_data.csv")

# Defining features (X) and target (y)
print('Performance Metrics For Motor_UPDRS')
X = df.drop(columns=['subject#', 'age' , 'sex', 'test_time', 'motor_updrs', 'total_updrs'])
y = df['motor_updrs']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

# Building and training the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Displaying coefficients
print("Intercept: ", model.intercept_)
print("Coefficient: ", model.coef_)

# Showing the predicted values
df_pred = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
print(df_pred)

# Performance metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
y_max = y_test.max()
y_min = y_test.min()
rmse_norm = rmse / (y_max - y_min)
r2 = metrics.r2_score(y_test, y_pred)
adjusted_r2 = 1 - (1-r2)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)

print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("RMSE (Normalised): ", rmse_norm)
print("R^2: ", r2)
print("Adjusted R^2: ", adjusted_r2)

# BASELINE MODEL
print("\n\n##### BASELINE MODEL #####")
y_base = np.mean(y_train)
y_pred_base = [y_base] * len(y_test)

# Showing the predicted values for the baseline model
df_base_pred = pd.DataFrame({"Actual": y_test, "Predicted": y_pred_base})
print(df_base_pred)

# Performance metrics for the baseline model
base_mae = metrics.mean_absolute_error(y_test, y_pred_base)
base_mse = metrics.mean_squared_error(y_test, y_pred_base)
base_rmse = math.sqrt(base_mse)
base_rmse_norm = base_rmse / (y_max - y_min)
base_r2 = metrics.r2_score(y_test, y_pred_base)
base_adjusted_r2 = 1 - (1-base_r2)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)

print("Baseline MAE: ", base_mae)
print("Baseline MSE: ", base_mse)
print("Baseline RMSE: ", base_rmse)
print("Baseline RMSE (Normalised): ", base_rmse_norm)
print("Baseline R^2: ", base_r2)
print("Baseline Adjusted R^2: ", base_adjusted_r2)
