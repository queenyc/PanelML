from sklearn.preprocessing import minmax_scale, scale
from scipy.stats import mstats

# winsorize by 3%
def winsorize_series(group):
    return mstats.winsorize(group, limits=[0.03, 0.97])
#df = df.groupby(level='DATE').transform(winsorize_series)

# minmax scaler by sklearn, collist scaled by all sample, collist scaled by date
def minmax_groupby(df, collist_all=[], collist_groupby=[]):
    if len(collist_all) > 0:
        for x in collist_all:
            df.loc[:, x] = df[x].transform(lambda x: minmax_scale(x.astype(float)))
    if len(collist_groupby) > 0:
        for x in collist_groupby:
            df.loc[:, x] = df[x].groupby(level=0).transform(lambda x: minmax_scale(x.astype(float)))
    return df


# minmax scaler by sklearn, collist scaled by all sample, collist scaled by date
def scaler_groupby(df, collist_all=[], collist_groupby=[]):
    if len(collist_all) > 0:
        for x in collist_all:
            df.loc[:, x] = df[x].transform(lambda x: scale(x.astype(float)))
    if len(collist_groupby) > 0:
        for x in collist_groupby:
            df.loc[:, x] = df[x].groupby(level=0).transform(lambda x: scale(x.astype(float)))
    return df


