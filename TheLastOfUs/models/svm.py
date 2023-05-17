from sklearn import svm


class SVM:

    def svm(self):
        """
         - Support Vector Machine algorithm
         - Returns prediction on X_test
        """
        # Create SVM classifier with a linear kernel
        # kernel='poly'
        # kernel='rbf', gamma='auto'
        clf = svm.SVC(kernel='linear')

        # Train the classifier on X_train
        clf.fit(self.datasets.X_train, self.datasets.y_train.values.ravel())

        # Predict labels on X_test
        y_pred = clf.predict(self.datasets.X_test)

        return y_pred
