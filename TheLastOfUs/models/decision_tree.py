from sklearn.tree import DecisionTreeClassifier
from grid_search_CV import grid_search_cv

# Best parameters from manual testing and grid search
params = {
    "random_state":42,
    "max_depth":5,
    "min_samples_leaf":30,
    "min_samples_split":20,
    "criterion":'entropy'    
}

# Arguments for grid search function

param_space = {
    "min_samples_leaf": [20, 30, 31, 40, 50],
    "min_samples_split": [2, 3, 10, 30],
    "max_depth": [5, 10, 18, 20],
    "max_features": [60, 80, 100]
}

model = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42)

def DT(data):
    # Get best parameters from grid search
    # params = grid_search_cv(data.X_train, data.y_train, param_space, model)

    # Initialize a DecisionTreeClassifier object
    clf = DecisionTreeClassifier(**params)

    # Fit the classifier to the training data, using top 5 features
    # to get the best results for accuracy and F1 and decrease computation time
    clf.fit(data.X_train.iloc[:, :5], data.y_train)

    # Use the trained classifier to make predictions on the test data
    y_pred = clf.predict(data.X_test.iloc[:, :5])

    # # Calculate the accuracy of the classifier on the test data
    # accuracy = accuracy_score(data.y_test, y_pred)

    # # Print the accuracy
    # print(f"Accuracy: {accuracy}")

    return y_pred