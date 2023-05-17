# Class storing the models for the data
import pandas as pd


class Datasets:

    def __init__(self, X_train, X_test, X_val, y_train, y_test, y_val):
        """
         - Input:
            - Paths for the 6 data sets
        """
        self.X_train = pd.read_csv(X_train)
        self.X_test = pd.read_csv(X_test)
        self.X_val = pd.read_csv(X_val)
        self.y_train = pd.read_csv(y_train)
        self.y_test = pd.read_csv(y_test)
        self.y_val = pd.read_csv(y_val)
