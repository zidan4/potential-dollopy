
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("drive/My Drive/ML_Process_Data_Files/Section_12_Model_Evaluation/housing_data.csv")

df.head()


features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
       'floors', 'waterfront', 'view', 'condition', 'sqft_above',
       'sqft_basement', 'yr_built', 'yr_renovated']

target = "price"

"""## Cross-Validation

Next, we'll want to split our data into training and testing sets:
"""

from sklearn.model_selection import train_test_split

y = df[target]
X = df[features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)
y_preds = model.predict(X_test)

"""## Evaluation Metrics

The most common evaluation metrics are r2_score, rmse and mae. Here, sklearn has its own implementation:
"""

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

r2 = r2_score(y_test, y_preds)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)

print("r2_score: {0}".format(r2))
print("rmse: {0}".format(rmse))
print("mae: {0}".format(mae))

"""## R2 Score

Intuition behind R-squared is that it tells us what percent of the prediction error in the y variable is eliminated/explained by your model. We use this to determine “goodness of fit.”

Here's an implementation of R2 so you can see the inner workings of the metric:
"""

def r2_score(y_test, y_preds):
    SS_reg = np.sum((y_test - y_preds)**2)
    SS_total = np.sum((y_test - np.mean(y_test))**2)
    r2 = 1-SS_reg/SS_total
    return r2

r2_score(y_test, y_preds)

"""## Adjusted R-Squared

The problem is that R-Squared can be easily hacked. If we overfit our model, this will always increase our r2 score. So the solution is to use adjusted R-squared. Adjusted R-squared will adjust our R-squared number based on the number of features in our model":
"""

def adj_r2_score(X, y_test, y_preds):
    SS_reg = np.sum((y_test - y_preds)**2)
    SS_total = np.sum((y_test - np.mean(y_test))**2)
    r2 = 1-SS_reg/SS_total

    N = len(X)
    p = len(X.columns)

    adj_r2 = 1-((1-r2)*(N-1))/(N-p-1)
    return adj_r2

adj_r2_score(X, y_test, y_preds)

"""## Mean Absolute Error

In simple terms, we’re just looking at the absolute average errors for each data point. Then taking an average. This gives us the magnitude of the average error in our dataset:
"""

def mean_absolute_error(y_test, y_preds):
    return np.sum(abs(y_preds - y_test))/len(y_preds)

mean_absolute_error(y_test, y_preds)

"""## Root Mean Squared Error

Instead of taking the absolute value of the errors, in this case we square the errors first. This forces all the errors to be positive. We take the average of the squared errors, which becomes mean-squared error. Then we take the square root, to get RMSE.
"""

def mean_squared_error(y_test, y_preds):
    return np.sum((y_preds - y_test)**2)/len(y_preds)

np.sqrt(mean_squared_error(y_test, y_preds))