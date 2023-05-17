# Gradient Boost Classifier

# set up libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

class GradientBoost:

    def __init__(self, datasets):
        self.datasets = datasets

    def gbc(self):
        """
         - Gradient Boost
         - Returns:
            - y predictions
        """
        # flatten y datasets
        self.datasets.y_train = self.datasets.y_train.to_numpy().flatten()
        self.datasets.y_val = self.datasets.y_val.to_numpy().flatten()
        self.datasets.y_test = self.datasets.y_test.to_numpy().flatten()
        
        # train model
        model = GradientBoostingClassifier(n_estimators=10000, learning_rate=0.1, 
                                           max_depth=2, random_state=42)
        model.fit(self.datasets.X_train, self.datasets.y_train)
        y_pred_train = model.predict(self.datasets.X_train)
        y_pred_val = model.predict(self.datasets.X_val)
        y_pred_test = model.predict(self.datasets.X_test)

        return y_pred_train, y_pred_val, y_pred_test