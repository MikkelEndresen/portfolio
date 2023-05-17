from datasets import Datasets
from logistic_regression import LogReg
from model_evaluation import Evaluation
import numpy as np
import decision_tree
import random_forest
from base_model import Base
from naive_bayes import NaiveBayes
from neural_net import NeuralNet
from gradient_boost import GradientBoost
from sklearn.metrics import confusion_matrix


pth = "../"
data = Datasets(pth + "X_train.csv", pth + "X_test.csv", pth + "X_val.csv",
                pth + "y_train.csv", pth + "y_test.csv", pth + "y_val.csv")
# two classes (Injury and Fatal combined into class Injury or Fatal)
data_b = Datasets(pth + "X_train.csv", pth + "X_test.csv", pth + "X_val.csv",
                  pth + "y_train_binary.csv", pth + "y_test_binary.csv", pth + "y_val_binary.csv")

log = LogReg(data)
eva = Evaluation()


# base model (predicts "Property Damage Only" for everything)
bm = Base(data_b)
bm_y_pred_train, bm_y_pred_val, bm_y_pred_test = bm.base()
eva = Evaluation()
print("Accuracy:", eva.accuracy(bm_y_pred_test, data_b.y_test))
print("Recall:", eva.recall(bm_y_pred_test, data_b.y_test))
print("Precision:", eva.precision(bm_y_pred_test, data_b.y_test))
print("F1 Score:", eva.f1_score(bm_y_pred_test, data_b.y_test))
print(np.count_nonzero(data_b.y_test == "Injury or Fatal"))
eva_conf = eva.confusion(bm_y_pred_test, data_b.y_test)
eva.plot_confusion(eva_conf)

# Naive Bayes
nb = NaiveBayes(data_b)
nb_y_pred_train, nb_y_pred_val, nb_y_pred_test = nb.nbc()
eva = Evaluation()
print("Accuracy:", eva.accuracy(nb_y_pred_test, data_b.y_test))
print("Recall:", eva.recall(nb_y_pred_test, data_b.y_test))
print("Precision:", eva.precision(nb_y_pred_test, data_b.y_test))
print("F1 Score:", eva.f1_score(nb_y_pred_test, data_b.y_test))
print(np.count_nonzero(data_b.y_test == "Injury or Fatal"))
eva_conf = eva.confusion(nb_y_pred_test, data_b.y_test)
eva.plot_confusion(eva_conf)

# Neural Net
nn = NeuralNet(data_b)
nn_y_pred_train, nn_y_pred_val, nn_y_pred_test = nn.nnc()
eva = Evaluation()
print("Accuracy:", eva.accuracy(nn_y_pred_test, data_b.y_test))
print("Recall:", eva.recall(nn_y_pred_test, data_b.y_test))
print("Precision:", eva.precision(nn_y_pred_test, data_b.y_test))
print("F1 Score:", eva.f1_score(nn_y_pred_test, data_b.y_test))
print(np.count_nonzero(data_b.y_test == "Injury or Fatal"))
eva_conf = eva.confusion(nn_y_pred_test, data_b.y_test)
eva.plot_confusion(eva_conf)

# Gradient Boosting
gb = GradientBoost(data_b)
gb_y_pred_train, gb_y_pred_val, gb_y_pred_test = gb.gbc()
eva = Evaluation()
print("Accuracy:", eva.accuracy(gb_y_pred_test, data_b.y_test))
print("Recall:", eva.recall(gb_y_pred_test, data_b.y_test))
print("Precision:", eva.precision(gb_y_pred_test, data_b.y_test))
print("F1 Score:", eva.f1_score(gb_y_pred_test, data_b.y_test))
print(np.count_nonzero(data_b.y_test == "Injury or Fatal"))
eva_conf = eva.confusion(gb_y_pred_test, data_b.y_test)
eva.plot_confusion(eva_conf)

# Decision Tree
DT_y_pred = decision_tree.DT(data_b)
eva = Evaluation()
print("Accuracy:", eva.accuracy(DT_y_pred, data_b.y_test))
print("Recall:", eva.recall(DT_y_pred, data_b.y_test))
print("Precision:", eva.precision(DT_y_pred, data_b.y_test))
print("F1 Score:", eva.f1_score(DT_y_pred, data_b.y_test))
print("Test non-property damage:", np.count_nonzero(data_b.y_test == "Injury or Fatal"))
eva_conf = eva.confusion(DT_y_pred, data_b.y_test)
eva.plot_confusion(eva_conf)

# Random Forest
RF_y_pred = random_forest.RF(data_b)
eva = Evaluation()
print("Accuracy:", eva.accuracy(RF_y_pred, data_b.y_test))
print("Recall:", eva.recall(RF_y_pred, data_b.y_test))
print("Precision:", eva.precision(RF_y_pred, data_b.y_test))
print("F1 Score:", eva.f1_score(RF_y_pred, data_b.y_test))
print("Test non-property damage:", np.count_nonzero(data_b.y_test == "Injury or Fatal"))
eva_conf = eva.confusion(RF_y_pred, data_b.y_test)
eva.plot_confusion(eva_conf)

# One vs Rest Logistic regression prediction
'''
log_y_pred = log.one_v_rest()

print(eva.accuracy(log_y_pred, data.y_test))
print(np.count_nonzero(data.y_test == "Fatal"))
eva_conf = eva.confusion(log_y_pred, data.y_test)
print(eva_conf)
eva.plot_confusion(eva_conf)
'''

'''
ecoc_y_pred = log.ecoc()
print(eva.accuracy(ecoc_y_pred, data.y_test))
eva_conf = eva.confusion(ecoc_y_pred, data.y_test)
eva.plot_confusion(eva_conf)
'''
