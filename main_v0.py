from panelMLlib.featnormalize import minmax_groupby
import pandas as pd

# read data
df = pd.read_csv('/Users/queenyc/Documents/Pyprojects/Data/OnlineNewsPopularity.csv')
df['Date'] = pd.to_datetime(df.url.apply(lambda x: x[20:30]))
df.columns = df.columns.str.strip()
df.set_index(['Date', 'timedelta'], inplace=True)

# define target
df['target'] = pd.qcut(df.shares, 3, labels=[0, 1, 2])
target = 'target'

# define features
featureslist = [x for x in df.columns if x not in ['url', target]]
df = minmax_groupby(df, collist_groupby=featureslist)

# Random forest
