import pandas as pd
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Read dataset into a DataFrame
df = pd.read_csv("E:\Foundation of Data Science\po2_data.csv")

# Log-transform the numeric columns
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = np.log1p(df[col])

# Defining features (X) and target (y)
X = df.drop(columns=['subject#', 'age', 'sex', 'test_time', 'motor_updrs', 'total_updrs'])
y = df['motor_updrs']

# Display a heatmap for collinearity check
plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm', center=0)
plt.show()

# Checking for multicollinearity with VIF
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)

# You may decide to drop columns with high VIF values (typically VIF > 10) or ones that make logical sense to remove based on domain knowledge.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Compute metrics for linear regression model after enhancements
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
r2 = metrics.r2_score(y_test, y_pred)
adjusted_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
nrmse = rmse / (y_test.max() - y_test.min())

print("\nImproved Multiple Linear Regression Metrics:")
print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("NRMSE: ", nrmse)  
print("R^2: ", r2)
print("Adjusted R^2: ", adjusted_r2)
