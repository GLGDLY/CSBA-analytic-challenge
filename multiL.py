from json import loads
import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm
import plotly.graph_objects as go

with open("./datas/dataset1_stats.txt", "r", encoding="utf-8") as f:
    datas = f.read().split("\n")
    headline_monthly_9ESG = loads(datas[12])
stat_df = pd.DataFrame(headline_monthly_9ESG).T
stat_df['max'] = stat_df.idxmax(axis="columns")

HSI_df = pd.read_csv("./datas/^HSI_weekly.csv")

fig = go.Figure(data=[go.Candlestick(x=HSI_df['Date'],
                open=HSI_df['Open'],
                high=HSI_df['High'],
                low=HSI_df['Low'],
                close=HSI_df['Close'])])
fig.show()

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