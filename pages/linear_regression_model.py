import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

@st.cache_resource
def lr_build(X, y):
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    # Initialize the linear regression model
    linear_model = LinearRegression()

    # Define the parameter grid for GridSearchCV
    param_grid = {
        'fit_intercept': [True, False],
        'copy_X': [True, False],
        'n_jobs': [None, -1],
        'positive': [True, False]
    }

    # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator=linear_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

    # Perform the grid search on the training data
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters found by GridSearch
    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)

    # Use the best model from GridSearch
    best_linear_model = grid_search.best_estimator_

    # Fit the best model to the training data
    best_linear_model.fit(X_train, y_train)

    # Make predictions on the train & test set
    x_pred = best_linear_model.predict(X_train)
    y_pred = best_linear_model.predict(X_test)

    # Save the best model to a .pkl file using joblib
    joblib.dump(best_linear_model, 'best_linear_regression_model.pkl')

    return y_pred, y_test, x_pred, y_train, best_params