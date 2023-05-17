# Logistic Regression algorithm

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn import svm


class LogReg:

    def __init__(self, datasets):
        self.datasets = datasets

    def one_v_rest(self):
        """
         - One vs Rest logistic regression
         - Returns:
            - y prediction on X_test
        """
        # Create a one-vs-all logistic regression model and fit it to the training data
        # Saga is recommended for larger datasets and uses a version of SGD
        lr = LogisticRegression(
            max_iter=1000, solver="saga", multi_class='ovr', class_weight={0: 75, 1: 4, 2: 1})
        # , class_weight={0: 31, 1: 2, 2: 1}
        # , penalty="l2", C=0.1,
        ovr = OneVsRestClassifier(lr)
        ovr.fit(self.datasets.X_train, self.datasets.y_train)

        # Make predictions on the testing data
        y_pred = ovr.predict(self.datasets.X_test)

        return y_pred

    def ecoc(self):
        lr = LogisticRegression(
            max_iter=1000, solver="saga", multi_class='ovr')
        # , class_weight={0: 31, 1: 2, 2: 1}
        # , penalty="l2", C=0.1,
        ovr = OneVsRestClassifier(lr)
        ovr.fit(self.datasets.X_train, self.datasets.y_train)

        # Convert the target labels to binary form
        lb = LabelBinarizer()
        y_train_binary = lb.fit_transform(self.datasets.y_train)

        # Train the classifier
        ovr.fit(self.datasets.X_train, y_train_binary)

        # Make predictions on the test set
        y_pred_binary = ovr.predict(self.datasets.X_test)

        # Convert the binary predictions back to multi-class labels
        y_pred = lb.inverse_transform(y_pred_binary)

        return y_pred

    # TODO: Try an SVM?
    def svm(self):
        # Create SVM classifier with a linear kernel
        clf = svm.SVC(kernel='linear')

        # Train the classifier on the training data
        clf.fit(self.datasets.X_train, self.datasets.y_train.values.ravel())

        # Predict labels on the test data
        y_pred = clf.predict(self.datasets.X_test)
        return y_pred
