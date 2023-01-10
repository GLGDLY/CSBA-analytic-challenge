#produce HSI along time,correlation tables, and pairwise table
#python3.10 -u plotCTnpairwise.py
from json import loads
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


with open("./datas/dataset1_stats.txt", "r", encoding="utf-8") as f:
    datas = f.read().split("\n")
    headline_monthly_9ESG = loads(datas[12])
stat_df = pd.DataFrame(headline_monthly_9ESG).T
stat_df['max'] = stat_df.idxmax(axis="columns")

HSI_df = pd.read_csv("./datas/HSI_monthly.csv")


fig = go.Figure(data=[go.Candlestick(x=HSI_df['Date'],
                open=HSI_df['Open'],
                high=HSI_df['High'],
                low=HSI_df['Low'],
                close=HSI_df['Close'])])
fig.show()

#append open price to stat_df
HSI_df = HSI_df.set_index(stat_df.index)
stat_df['Open Price'] = HSI_df.iloc[:,1].tolist()

#create correlation table
print(stat_df.corr())
stat_df.corr().to_csv('./datas/dataset1_correlation_table.csv')


#Creating a pairwise plot in Seaborn, colored by year
"""
stat_df['Date'] = HSI_df.iloc[:,0].tolist()
stat_df['Date'] = pd.to_datetime(stat_df['Date'])
stat_df['Year'] = stat_df['Date'].dt.year
temp = sns.pairplot(stat_df, hue='Year')
"""
temp = sns.pairplot(stat_df, diag_kind="kde")
temp.map_lower(sns.regplot, robust=True)
temp.savefig("./datas/pairwise3.png")

#Creating a pairwise plot in Pairgrid

# Function to calculate correlation coefficient between two arrays
def corr(x, y, **kwargs):
    # Calculate the value
    coef = np.corrcoef(x, y)[0][1]
    # Make the label
    label = r'$\rho$ = ' + str(round(coef, 2))
    # Add the label to the plot
    ax = plt.gca()
    ax.annotate(label, xy = (0.2, 0.95), size = 20, xycoords = ax.transAxes)
    
# Create a pair grid instance
grid = sns.PairGrid(data= stat_df, vars=['Climate Change', 'Pollution & Waste', 'Corporate Governance',
       'Natural Capital', 'Product Liability', 'Human Capital',
       'Business Ethics & Values', 'Community Relations', 'Open Price'])

# Map the plots to the locations
grid = grid.map_upper(plt.scatter)
grid = grid.map_upper(corr)
grid = grid.map_lower(sns.kdeplot)
grid = grid.map_diag(plt.hist)
grid.savefig("./datas/gridpairwise.png")


