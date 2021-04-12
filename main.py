import pandas
from sklearn.linear_model import LinearRegression

# read data from csv
df = pandas.read_csv("Auto.csv")
X = df[['horsepower']]
y = df['mpg']
reg = LinearRegression().fit(X, y)
reg.score(X, y)
reg.coef_
reg.intercept_