# python3.10 -u test.py
"""
from vega_datasets import data
stocks = data.stocks()
print(stocks)


import altair as alt
alt.Chart(stocks).mark_line().encode(
  x='date:T',
  y='price',
  color='symbol'
).interactive(bind_y=False)
"""
import altair as alt
from altair.expr import datum

from vega_datasets import data
stocks = data.stocks.url

base = alt.Chart(stocks).encode(
    x='date:T',
    y='price:Q',
    color='symbol:N'
).transform_filter(
    datum.symbol == 'GOOG'
)

output = base.mark_line() + base.mark_point()
output.show()