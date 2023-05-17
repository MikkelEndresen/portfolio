# Car Crash Severity Prediction - The Last of Us

1. Run `feature_engineering.py`. This script will open the original datasets, drop unnecessary features, encode other features into the necessary format, and split the data into training, validation, and test sets. The split datasets are saved as separate `.csv` files in the Data folder. The target variable is Crash Severity, which has three classes: 'Property Damage Only', 'Injury', and 'Fatal'. 'Injury' and 'Fatal' were combined into one class 'Injury or Fatal'. The `feature_selection.py` contains a function that returns the top n (default 178) most correlated features to the target variable in descending order (highest to lowest correlation).

- ACT Road Crash Data (CC - Attribution 4.0 International): https://www.data.act.gov.au/Transport/ACT-Road-Crash-Data/6jn4-m8rx
- Traffic Speed Camera Locations (CC - Attribution 4.0 International): https://www.data.act.gov.au/Justice-Safety-and-Emergency/Traffic-speed-camera-locations/426s-vdu4

2. The Models folder contains all the machine learning models that were created. Hyperparameters can be changed inside these files.

- `ada_boost.py`
- `base_model.py`
- `decision_tree.py`
- `gradient_boost.py`
- `logistic_regression.py`
- `naive_bayes.py`
- `neural_net.py`
- `random_forest.py`
- `svm.py`

3. The Models folder also contains helper code.

- `datasets.py` - creates a class that loads the split datasets into variables
- `grid_search_CV.py` - creates a function to do grid search with cross validation and return the best-performing parameters
- `model_evaluation.py` - creates a class containing the necessary functions for evaluating the performance of a model
- `test.py` - loads each model and performs evaluation
- `final_model.py` - loads and runs the final model with the best metrics
