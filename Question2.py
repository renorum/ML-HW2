import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import r2_score
import statsmodel.api as sm
import statsmodel.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from yellowbrick.datasets import load_concrete
from yellowbrick.regressor import ResidualsPlot

df = pd.read_csv("Auto copy.csv", ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year', 'origin'])
pd.plotting.scatter_matrix(df)

corrMatrix = df.corr()

# independent variables
X = df[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year', 'origin']]
# dependent variable
y = df['mpg']

# create the linear regression model
regr = linear_model.LinearRegression()
model = regr.fit(X, y)

# get predictions
mpgPred = model.predict(X)

# prints matrix of coefficients in the order fed to the model
print(regr.coef_)
print(regr.intercept_)
print(r2_score(y, mpgPred))

data = sm.datasets.df.load_pandas()
results = smf.ols('mpg ~ cylinders + displacement + horsepower + weight + acceleration + year + origin', data=data.data).fit()
sm.graphics.influence_plot(results)
plt.show()

X, y = load_concrete()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
Imodel = Ridge()
visualizer = ResidualsPlot(Imodel)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()