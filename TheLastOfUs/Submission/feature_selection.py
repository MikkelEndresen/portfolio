import pandas as pd

def filter_features_pearson(n):
    X_train = pd.read_csv("X_train.csv")
    X_val = pd.read_csv("X_val.csv")
    X_test = pd.read_csv("X_test.csv")
    y_train = pd.read_csv("y_train_binary.csv")
    # def feature_select_pearson(train, test):
    print('Filter method to select top n features...')
    train = X_train
    val = X_val
    test = X_test
    target = "CRASH_SEVERITY"
    train[target] = y_train[target]
    train[target].replace(('Property Damage Only', 'Injury or Fatal'), (1, 0), inplace=True)
    features = train.columns.tolist()
    # features.remove("CRASH_ID")
    features.remove(target)
    featureSelect = features[:]
    # print(features) # 178 features

    print(featureSelect)
    # remove features with a missing value ratio greater than 0.99
    for feature in features:
        if train[feature].isnull().sum() / train.shape[0] >= 0.99:
            featureSelect.remove(feature)

    # calculate the pearson correlation
    corr = []
    for feature in featureSelect:
        print(train[[feature, target]])
        corr.append(abs(train[[feature, target]].fillna(0).corr().values[0][1]))

    # get top n features with the highest correlation
    se = pd.Series(corr, index=featureSelect).sort_values(ascending=False)
    feature_select = se[:n].index.tolist()
    print(feature_select)
    print(len(feature_select))
    print('done')
    # print(train[feature_select + [target]], test[feature_select + [target]])
    return train[feature_select], val[feature_select], test[feature_select]

if __name__ == '__main__':
    filter_features_pearson()