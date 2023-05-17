# Base model that predicts "Property Damage Only" for all datapoints

import numpy as np


class Base:

    def __init__(self, datasets):
        self.datasets = datasets

    def base(self):
        """
         - Base model
         - Returns:
            - y prediction as "Property Damage Only"
        """
        y_pred_test = np.full(len(self.datasets.y_test), 'Property Damage Only')
        y_pred_val = np.full(len(self.datasets.y_val), 'Property Damage Only')
        y_pred_train = np.full(len(self.datasets.y_train), 'Property Damage Only')

        return y_pred_train, y_pred_val, y_pred_test