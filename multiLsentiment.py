#run multi-linear model 
#python3.10 -u multiLsentiment.py
from sklearn import linear_model
import statsmodels.api as sm
from json import loads
import pandas as pd

stat_df = pd.read_csv('./datas/dataset1_per_class_ss.csv')

HSI_df = pd.read_csv("./datas/HSI_monthly.csv")
#append open price to stat_df
HSI_df = HSI_df.set_index(stat_df.index)
stat_df['Open Price'] = HSI_df.iloc[:,1].tolist()
print(stat_df)


#regression analysis
x = stat_df[['Climate Change', 'Pollution & Waste', 'Corporate Governance',
       'Natural Capital', 'Product Liability', 'Human Capital',
       'Business Ethics & Values', 'Community Relations']]
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