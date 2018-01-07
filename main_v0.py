from panelMLlib.processdata import minmax_groupby, splitdata
from panelMLlib.mlfunctions import *
import pandas as pd

if __name__ == '__main__':

    # read data
    df = pd.read_csv('/Users/queenyc/Documents/Pyprojects/Data/OnlineNewsPopularity.csv')
    df['Date'] = pd.to_datetime(df.url.apply(lambda x: x[20:30]))
    df.columns = df.columns.str.strip()
    df.set_index(['Date', 'timedelta'], inplace=True)
    df.sort_index(inplace=True)

    # define target
    df['target'] = pd.qcut(df.shares, 3, labels=[0, 1, 2])
    target = 'target'

    # define features
    featureslist = [x for x in df.columns if x not in ['url', 'timedelta', target]]
    df = minmax_groupby(df, collist_groupby=featureslist)

    # split train and valid data
    train_data, valid_data, test_data, train_label, valid_label, test_label = splitdata(df, featureslist, target,
                                                                                        '2013-1-1', '2014-6-1',
                                                                                        '2014-6-1', '2015-6-1',
                                                                                        '2014-6-1', '2015-1-1')
    #rf
    rf = rf(train_data, train_label)
    rf_mainfeatures(rf, train_data, n=10)

    # ml model
    d1, d2, d3 = ml_predict(rf, train_data, valid_data, test_data,
                            train_label, valid_label, test_label)
    _, _, _ = ml_predictprob(rf, train_data, valid_data, test_data,
                             train_label, valid_label, test_label)

