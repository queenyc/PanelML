import pandas as pd

# rf: params for training data
def rf(data_features, data_label):
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_jobs=1,
                                criterion='entropy',
                                n_estimators=100,
                                max_features='sqrt',
                                min_samples_split=200,
                                bootstrap=True,
                                class_weight='balanced')
    rf.fit(data_features, data_label.values.ravel())
    return rf


def rf_mainfeatures(mlmodel, train_data, n=10):
    from collections import Counter
    main_features = Counter(dict(zip(train_data, mlmodel.feature_importances_)))
    print(main_features.most_common(n))

############################################################

# ml model evaluation

# ml accuracy score
def ml_accuracy(train_label, train_predict, valid_label, valid_predict, test_label, test_predict):
    from sklearn.metrics import accuracy_score
    print('Train accuracy : ', accuracy_score(train_label, train_predict))
    print('Valid accuracy : ', accuracy_score(valid_label, valid_predict))
    print('Test accuracy : ', accuracy_score(test_label, test_predict))


# ml log loss
def ml_logloss(train_label, train_prob, valid_label, valid_prob, test_label, test_prob):
    from sklearn.metrics import log_loss
    print('Train logloss : ', log_loss(train_label, train_prob))
    print('Valid logloss : ', log_loss(valid_label, valid_prob))
    print('Test logloss : ', log_loss(test_label, test_prob))


# ml predict class
def ml_predict(mlmodel, train_data, valid_data, test_data, train_label, valid_label, test_label):
    train_predict = mlmodel.predict(train_data)
    valid_predict = mlmodel.predict(valid_data)
    test_predict = mlmodel.predict(test_data)
    ml_accuracy(train_label, train_predict, valid_label, valid_predict, test_label, test_predict)
    return pd.DataFrame(train_predict, index=train_data), pd.DataFrame(valid_predict, index=valid_data), \
           pd.DataFrame(test_predict, index=test_data)


# ml class prob
def ml_predictprob(mlmodel, train_data, valid_data, test_data, train_label, valid_label, test_label):
    train_prob = mlmodel.predict_proba(train_data)
    valid_prob = mlmodel.predict_proba(valid_data)
    test_prob = mlmodel.predict_proba(test_data)
    ml_logloss(train_label, train_prob, valid_label, valid_prob, test_label, test_prob)
    return pd.DataFrame(train_prob, index=train_data), pd.DataFrame(valid_prob, index=valid_data), \
           pd.DataFrame(test_prob, index=test_data)







