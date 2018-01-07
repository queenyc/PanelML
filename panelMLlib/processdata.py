from sklearn.preprocessing import minmax_scale, scale
from scipy.stats import mstats
import pandas as pd


# winsorize by 3%
def winsorize_series(group):
    return mstats.winsorize(group, limits=[0.03, 0.97])
#df['a'] = df.a.groupby(level=0).transform(winsorize_series)


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


# define train, valid and test dataset
def splitdata(df, featureslist, col_target, train_start, train_end, valid_start, valid_end, test_start='', test_end=''):

    train_data = df.loc[pd.IndexSlice[train_start: train_end, :], featureslist]
    valid_data = df.loc[pd.IndexSlice[valid_start: valid_end, :], featureslist]
    test_data = df.loc[pd.IndexSlice[test_start: test_end, :], featureslist]
    train_label = df.loc[pd.IndexSlice[train_start: train_end, :], col_target]
    valid_label = df.loc[pd.IndexSlice[valid_start: valid_end, :], col_target]
    test_label = df.loc[pd.IndexSlice[test_start: test_end, :], col_target]

    return train_data, valid_data, test_data, train_label, valid_label, test_label

