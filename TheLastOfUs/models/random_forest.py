from sklearn.ensemble import RandomForestClassifier
from grid_search_CV import grid_search_cv

# These are the best parameters based off grid search with cross validation
params = {
    'n_estimators': 80,
    'max_depth': 20,
    'max_features': 60,
    'min_samples_leaf': 31,
    'min_samples_split': 2,
    'criterion': "gini",
    'n_jobs': 15,
    'random_state': 42,
}

# Arguments for grid search function

param_space = {
    "min_samples_leaf": [30, 31],
    "min_samples_split": [2, 10],
    "max_depth": [18, 20],
    "max_features": [60, 80]
}
# Takes very long
parameter_space = {
    "min_samples_leaf": [30, 31, 40],
    "min_samples_split": [2, 3, 10],
    "max_depth": [9, 10, 18, 20],
    "max_features": [60, 80, 100]
}

model = RandomForestClassifier(
    criterion="gini",
    n_jobs=15,
    random_state=42)

def RF(data):
    # Get best parameters from grid search
    # params = grid_search_cv(data.X_train, data.y_train, param_space, model)
    
    # Initialize a DecisionTreeClassifier object
    clf = RandomForestClassifier(**params)

    # Fit the classifier to the training data, using top 16 features
    # to get the best results for accuracy and F1 and decrease computation time
    clf.fit(data.X_train.iloc[:, :16], data.y_train)

    # Use the trained classifier to make predictions on the test data
    y_pred = clf.predict(data.X_test.iloc[:, :16])

    # # Calculate the accuracy of the classifier on the test data
    # accuracy = accuracy_score(data.y_test, y_pred)

    # # Print the accuracy
    # print(f"Accuracy: {accuracy}")

    return y_pred