#run multi-linear model 
#python3.10 -u multiL.py
from sklearn import linear_model
import statsmodels.api as sm
from json import loads
import pandas as pd

with open("./datas/dataset1_stats.txt", "r", encoding="utf-8") as f:
    datas = f.read().split("\n")
    headline_monthly_9ESG = loads(datas[12])
stat_df = pd.DataFrame(headline_monthly_9ESG).T
stat_df['max'] = stat_df.idxmax(axis="columns")

HSI_df = pd.read_csv("./datas/HSI_monthly.csv")
#append open price to stat_df
HSI_df = HSI_df.set_index(stat_df.index)
stat_df['Open Price'] = HSI_df.iloc[:,1].tolist()

#1st regression analysis
x = stat_df[['Climate Change', 'Pollution & Waste', 'Corporate Governance',
       'Natural Capital', 'Product Liability', 'Human Capital',
       'Business Ethics & Values', 'Community Relations']]
y = stat_df['Open Price']

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

#2nd iteration 
print("\n2nd iteration")
x = stat_df[['Climate Change', 'Pollution & Waste', 'Corporate Governance',
       'Natural Capital', 'Product Liability', 'Human Capital',
         'Community Relations']]
y = stat_df['Open price']

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

#3rd iteration 
print("\n3rd iteration")
x = stat_df[['Climate Change', 'Pollution & Waste', 'Corporate Governance',
        'Product Liability', 'Human Capital','Community Relations']]
y = stat_df['Open Price']

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

#4th iteration 
print("\n4th iteration\n")
x = stat_df[['Climate Change', 'Pollution & Waste', 
        'Product Liability', 'Human Capital','Community Relations']]
y = stat_df['Open Price']

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

#5th iteration 
print("\n5th iteration\n")
x = stat_df[['Climate Change', 'Pollution & Waste', 
        'Product Liability', 'Community Relations']]
y = stat_df['Open Price']

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
