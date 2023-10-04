import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

# Read dataset into a DataFrame
df = pd.read_csv("E:\Foundation of Data Science\po2_data.csv")

# Extract the specific features and the target column 'MV'
features = ['jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)', 
           'jitter(ddp)', 'shimmer(%)', 'shimmer(abs)', 'shimmer(apq3)', 
           'shimmer(apq5)', 'shimmer(apq11)', 'shimmer(dda)', 'nhr', 
           'hnr', 'rpde', 'dfa', 'ppe']
df = df[features]

# Separate explanatory variables (x) from the response variable (y)
x = df.iloc[:,:-1]
y = df.iloc[:,-1]

# Build and evaluate the linear regression model
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
print(model.summary())

"""Log transformation of the features"""

# Apply log transformation
for feature in features[:-1]:
    df["LOG_" + feature] = df[feature].apply(np.log)

# Drop the original features
df = df.drop(features[:-1], axis=1)

# Separate explanatory variables (x) from the response variable (y)
x = df.iloc[:,:-1]
y = df.iloc[:,-1]

# Build and evaluate the linear regression model after log transform
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
print(model.summary())
