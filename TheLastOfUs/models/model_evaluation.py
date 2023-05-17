import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score


class Evaluation:

    # TODO: add bias and variance

    def precision(self, y_pred, y_true):
        """
         - Multi-class evaluation
         - Input:
            - y_true
            - y_pred
         - Output:
            - returns precision score
        """
        # macro - unweighted average across all classes
        # Can use wieghted - weighted per num in each class
        return precision_score(y_true, y_pred, average="macro")

    def recall(self, y_pred, y_true):
        """
         - Multi-class evaluation
         - True Positives / (True Positives + False Negatives)
         - Input:
            - y_true
            - y_pred
         - Output:
            - returns recall  score
        """
        # macro - unweighted average across all classes
        # Can use wieghted - weighted per num in each class
        return recall_score(y_true, y_pred, average="macro")

    def f1_score(self, y_pred, y_true):
        """
         - Multi-class evaluation
         - 2 * (Precision * Recall) / (Precision + Recall)
         - Input:
            - y_true
            - y_pred
         - Output:
            - returns f_1 score
        """
        # macro - unweighted average across all classes
        # Can use wieghted - weighted per num in each class
        return f1_score(y_true, y_pred, average="macro")

    def confusion(self, y_pred, y_true):
        """
         - Input:
            - y_pred, e.g. [0, 2, 1, 2, 1, 0]
            - y_true, e.g. [1, 2, 0, 2, 1, 0]
        - Output:
            - Returns Confusion matrix
        """
        return confusion_matrix(y_pred, y_true)

    def plot_confusion(self, confusion):
        # Plot confusion matrix as heatmap
        sns.heatmap(confusion, annot=True, cmap="Blues")

        # Add axis labels and title
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")

        plt.show()

    def plot_roc(self, y_pred, y_true):
        """
        - Input:
            - binary classification only
            - y_pred of positive class
            - y_true
        - Output:
            - plots ROC curve with AUC score
        """

        # fpr - false positve rate
        # tpr - true positive rate
        # thresholds, between 0 and 1
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)

        # AUC score
        auc_score = roc_auc_score(y_true, y_prob)

        # plot the ROC curve
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score)
        plt.plot([0, 1], [0, 1], 'r--', label='Random classifier')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC curve')
        plt.legend(loc="lower right")
        plt.show()

    def auc_score(self, y_pred, y_true):
        """
        - binary classification only
        - Input:
            - y_pred of positive class
            - y_true
        - Output:
            - Returns AUC score between 0-1
        """
        return roc_auc_score(y_true, y_pred)

    def accuracy(self, y_pred, y_true):
        """
        - Multi-class 
        - Input:
            - y_pred
            - y_true
        - Output:
            - Returns accuracy
        """
        from sklearn.metrics import accuracy_score
        return accuracy_score(y_true, y_pred)


### PURELY FOR TESTING ###
### REMOVE BEFORE SUBMISSION ###
# generate a synthetic binary classification dataset
if __name__ == "__main__":
    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=5, random_state=42)

    # split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # train a logistic regression model on the training set
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # predict the probabilities on the testing set
    # [","]
    y_prob = model.predict_proba(X_test)[:, 1]
    y_predict = model.predict(X_test)
    evaluation = Evaluation()

    evaluation.plot_roc(y_prob, y_test)
    print(evaluation.auc_score(y_prob, y_test))
    evaluation.confusion(y_predict, y_test)
    print(evaluation.accuracy(y_predict, y_test))
    print(evaluation.recall(y_predict, y_test))
    print(evaluation.precision(y_predict, y_test))
    print(evaluation.f1_score(y_predict, y_test))
