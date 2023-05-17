# Neural Net Classifier

# set up libraries
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

class NeuralNet:

    def __init__(self, datasets):
        self.datasets = datasets

    def nnc(self):
        """
         - Neural Net
         - Returns:
            - y predictions
        """
        # flatten y datasets
        self.datasets.y_train = self.datasets.y_train.to_numpy().flatten()
        self.datasets.y_val = self.datasets.y_val.to_numpy().flatten()
        self.datasets.y_test = self.datasets.y_test.to_numpy().flatten()
        
        # train model
        model = MLPClassifier(hidden_layer_sizes=(100,50,25), activation='relu',
                              solver='adam', max_iter=500, random_state=42)
        model.fit(self.datasets.X_train, self.datasets.y_train)
        y_pred_train = model.predict(self.datasets.X_train)
        y_pred_val = model.predict(self.datasets.X_val)
        y_pred_test = model.predict(self.datasets.X_test)

        return y_pred_train, y_pred_val, y_pred_test