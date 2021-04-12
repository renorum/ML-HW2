import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score

df = pd.read_csv("Carseats.csv", ['Sales', 'Price', 'Urban', 'US'])

# independent variables
X = df[['Price', 'Urban', 'US']]
# dependent variable
y = df['Sales']

# create the linear regression model
regr = linear_model.LinearRegression()
model = regr.fit(X, y)

# get predictions
salesPred = model.predict(X)

# prints matrix of coefficients in the order fed to the model
print(regr.coef_)
print(regr.intercept_)
print(r2_score(y, salesPred))