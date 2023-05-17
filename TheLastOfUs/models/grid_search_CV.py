from sklearn.model_selection import GridSearchCV

def grid_search_cv(x_train, y_train, param_space, model):
    # do the grid search with cross validation
    print("Tuning hyper-parameters by gridsearch")
    grid = GridSearchCV(model, param_space, cv=5)
    grid.fit(x_train, y_train)

    # get the best parameters on the sample of training set
    print("The best parameters are: ", grid.best_params_)
    print("Best score is: ", grid.best_score_)
    return grid.best_params_