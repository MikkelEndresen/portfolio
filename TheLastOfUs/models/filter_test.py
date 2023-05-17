import pandas as pd
from datasets import Datasets
from model_evaluation import Evaluation
import numpy as np

pth = "../"
data = Datasets(pth + "X_train.csv", pth + "X_test.csv", pth + "X_val.csv",
                pth + "y_train_binary.csv", pth + "y_test_binary.csv", pth + "y_val_binary.csv")

# def feature_select_pearson(train, test):
print('Filter method to select top 16 features...')
train = data.X_train
val = data.X_val
test = data.X_test
target = "CRASH_SEVERITY"
train[target] = data.y_train[target]
train[target].replace(('Property Damage Only', 'Injury or Fatal'), (1, 0), inplace=True)
features = train.columns.tolist()
# features.remove("CRASH_ID")
features.remove(target)
featureSelect = features[:]
# print(features) # 178 features

# print(featureSelect)
# remove features with a missing value ratio greater than 0.99
for feature in features:
    if train[feature].isnull().sum() / train.shape[0] >= 0.99:
        featureSelect.remove(feature)

# calculate the pearson correlation
corr = []
for feature in featureSelect:
    # print(train[[feature, target]])
    corr.append(abs(train[[feature, target]].fillna(0).corr().values[0][1]))

# get top 16 features with the highest correlation
se = pd.Series(corr, index=featureSelect).sort_values(ascending=False)
feature_select = se[:16].index.tolist()
print(feature_select)
print('done')