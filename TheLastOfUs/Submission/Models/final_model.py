import numpy as np
from random_forest import RF
from model_evaluation import Evaluation
from datasets import Datasets

pth = "../Data/"
data_b = Datasets(pth + "X_train.csv", pth + "X_test.csv", pth + "X_val.csv",
                  pth + "y_train_binary.csv", pth + "y_test_binary.csv", pth + "y_val_binary.csv")


eva = Evaluation()

RF_y_pred = RF(data_b)
print("Accuracy:", eva.accuracy(RF_y_pred, data_b.y_test))
print("Recall:", eva.recall(RF_y_pred, data_b.y_test))
print("Precision:", eva.precision(RF_y_pred, data_b.y_test))
print("F1 Score:", eva.f1_score(RF_y_pred, data_b.y_test))
print("Test non-property damage:",
      np.count_nonzero(data_b.y_test == "Injury or Fatal"))
eva_conf = eva.confusion(RF_y_pred, data_b.y_test)
eva.plot_confusion(eva_conf)
