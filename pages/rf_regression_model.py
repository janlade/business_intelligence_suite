__author__ = "Jan Lade"
__copyright__ = "Copyright 2024, Jan Lade"
__credits__ = ["Jan Lade", "Tom Debus"]
__version__ = "1.0"
__maintainer__ = "Jan Lade"
__status__ = "Production"


#imports
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import joblib

@st.cache_resource
def rf_model(X, y, selected_file):
    """
    Train a Random Forest Regression model using GridSearchCV and return values for evaluation.

    Parameters:
    - X (DataFrame): The features for training.
    - y (DataFrame): The target variable for training.

    Returns:
    - y_pred (array): Predicted target values for the test set.
    - y_test (array): True target values for the test set.
    - x_pred (array): Predicted target values for the training set.
    - y_train (array): True target values for the training set.
    - best_params (dict): Best hyperparameters found by GridSearchCV.
    """    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    # Initialize the random forest regressor
    rf_model = RandomForestRegressor(random_state=42)

    # Define a grid of hyperparameters to search
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

    # Perform the grid search on the training data
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters found by GridSearch
    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)

    # Use the best model from GridSearch
    best_rf_model = grid_search.best_estimator_

    # Fit the best model to the training data
    best_rf_model.fit(X_train, y_train)

    # Make predictions on the train & test set
    x_pred = best_rf_model.predict(X_train)
    y_pred = best_rf_model.predict(X_test)

    # Save the best model to a .pkl file using joblib
    joblib.dump(best_rf_model, f'{selected_file}_best_random_forest_regressor_model.pkl')

    return y_pred, y_test, x_pred, y_train, best_params
