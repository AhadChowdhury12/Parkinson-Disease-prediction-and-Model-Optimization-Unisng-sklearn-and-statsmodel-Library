import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import PowerTransformer

# Read dataset into a DataFrame
df = pd.read_csv("E:\Foundation of Data Science\po2_data.csv")

# Separate explanatory variables (x) from the response variable (y)
x_original = df.drop(columns=['subject#', 'age' , 'sex', 'test_time', 'motor_updrs', 'jitter(rap)', 'jitter(ddp)', 'shimmer(apq3)', 'shimmer(dda)', 'total_updrs'])
y = df['motor_updrs']

# Build and evaluate the linear regression model
x_original = sm.add_constant(x_original)
model = sm.OLS(y, x_original).fit()
pred = model.predict(x_original)
model_details = model.summary()
print("Model Before Transformation:")
print(model_details)

"""
APPLY POWER TRANSFORMER TO EXPLANATORY VARIABLES ONLY
"""

# Drop the previously added constant
x_original = x_original.drop(["const"], axis=1)

# Create a Yeo-Johnson transformer
scaler = PowerTransformer(method='yeo-johnson')  # Fix the method argument

# Apply the transformer to make all explanatory variables more Gaussian-looking
std_x = scaler.fit_transform(x_original.values)

# Restore column names of explanatory variables
std_x_df = pd.DataFrame(std_x, index=x_original.index, columns=x_original.columns)

# Rebuild and reevaluate the linear regression model using the transformed explanatory variables
std_x_df = sm.add_constant(std_x_df)
model = sm.OLS(y, std_x_df).fit()
pred = model.predict(std_x_df)
model_details = model.summary()
print("\nModel After Transformation:")
print(model_details)
