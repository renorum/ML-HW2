import csv
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("Boston copy.csv")

# independent variables
X = df[['zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat', 'medv']]
# dependent variable
y = df['crim']

# create the linear regression model - all predictors against crim
regr = linear_model.LinearRegression()
model = regr.fit(X, y)

# get predictions
salesPred = model.predict(X)

# prints coefficients in the order fed to the model
print(regr.coef_)
print(regr.intercept_)
print(r2_score(y, salesPred))

# lin reg for each predictor against crim
for predictor in df:
    print(predictor)
    X = df[predictor]
    y = df['crim']
    reg = LinearRegression().fit(X, y)
    reg.score(X, y)
    reg.coef_
    reg.intercept_