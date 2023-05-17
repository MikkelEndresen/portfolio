# Naive Bayes Classifier

# set up libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB

class NaiveBayes:

    def __init__(self, datasets):
        self.datasets = datasets

    def undummify(self, df, prefix_sep="_"):
        # for Naive Bayes, we do not use one-hot encoding.
        # convert one-hot encoding to ordinal encoding-
        # https://stackoverflow.com/questions/50607740/reverse-a-get-dummies-encoding-in-pandas
        cols2collapse = {
            item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
        }
        series_list = []
        for col, needs_to_collapse in cols2collapse.items():
            if needs_to_collapse:
                undummified = (
                    df.filter(like=col)
                    .idxmax(axis=1)
                    .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                    .rename(col)
                )
                series_list.append(undummified)
            else:
                series_list.append(df[col])
        undummified_df = pd.concat(series_list, axis=1)
        return undummified_df

    def nbc(self):
        """
         - Naive Bayes
         - Returns:
            - y predictions
        """
        # flatten y datasets
        self.datasets.y_train = self.datasets.y_train.to_numpy().flatten()
        self.datasets.y_val = self.datasets.y_val.to_numpy().flatten()
        self.datasets.y_test = self.datasets.y_test.to_numpy().flatten()
        
        # turn one-hot encoded data back into categorical data (Naive Bayes needs each feature to be independent)
        self.datasets.X_train.rename({'CRASH_DATE_dayofweek': 'dayofweek', 'CRASH_TIME_hour': 'hour'}, axis=1, inplace=True)
        self.datasets.X_val.rename({'CRASH_DATE_dayofweek': 'dayofweek', 'CRASH_TIME_hour': 'hour'}, axis=1, inplace=True)
        self.datasets.X_test.rename({'CRASH_DATE_dayofweek': 'dayofweek', 'CRASH_TIME_hour': 'hour'}, axis=1, inplace=True)
        self.datasets.X_train = self.undummify(self.datasets.X_train)
        self.datasets.X_val = self.undummify(self.datasets.X_val)
        self.datasets.X_test = self.undummify(self.datasets.X_test) 
        
        # then encode categorical data as ordinal data 
        enc = OrdinalEncoder()
        enc.fit(self.datasets.X_train[["SUBURB", "CRASH", "LIGHTING", "ROAD", "WEATHER"]])
        self.datasets.X_train[["SUBURB", "CRASH", "LIGHTING", "ROAD", "WEATHER"]] = enc.transform(self.datasets.X_train[["SUBURB", "CRASH", "LIGHTING", "ROAD", "WEATHER"]])
        self.datasets.X_val[["SUBURB", "CRASH", "LIGHTING", "ROAD", "WEATHER"]] = enc.transform(self.datasets.X_val[["SUBURB", "CRASH", "LIGHTING", "ROAD", "WEATHER"]])
        self.datasets.X_test[["SUBURB", "CRASH", "LIGHTING", "ROAD", "WEATHER"]] = enc.transform(self.datasets.X_test[["SUBURB", "CRASH", "LIGHTING", "ROAD", "WEATHER"]])

        # bin distance (all NB data should be categorical)
        self.datasets.X_train["DISTANCE"] = pd.cut(self.datasets.X_train["DISTANCE"], bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],labels=[0,1,2,3,4,5,6,7,8,9])
        self.datasets.X_val["DISTANCE"] = pd.cut(self.datasets.X_val["DISTANCE"], bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],labels=[0,1,2,3,4,5,6,7,8,9])
        self.datasets.X_test["DISTANCE"] = pd.cut(self.datasets.X_test["DISTANCE"], bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],labels=[0,1,2,3,4,5,6,7,8,9])

        # train model
        model = CategoricalNB()
        model.fit(self.datasets.X_train, self.datasets.y_train)
        y_pred_train = model.predict(self.datasets.X_train)
        y_pred_val = model.predict(self.datasets.X_val)
        y_pred_test = model.predict(self.datasets.X_test)

        return y_pred_train, y_pred_val, y_pred_test