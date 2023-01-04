from json import loads
import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm

with open("dataset1_stats.txt", "r", encoding="utf-8") as f:
  datas = f.read().split("\n")
  headline_monthly_ESG = loads(datas[9])



"""
x = df[['interest_rate','unemployment_rate']]
y = df['index_price']

# with sklearn
regr = linear_model.LinearRegression()
regr.fit(x, y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# with statsmodels
x = sm.add_constant(x) # adding a constant
 
model = sm.OLS(y, x).fit()
predictions = model.predict(x) 
 
print_model = model.summary()
print(print_model)
"""