__author__ = "Jan Lade"
__copyright__ = "Copyright 2024, Jan Lade"
__credits__ = ["Jan Lade", "Tom Debus"]
__version__ = "1.0"
__maintainer__ = "Jan Lade"
__status__ = "Production"


#imports
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# sns and plt styling
sns.set_theme(style='darkgrid')
matplotlib.rcParams['font.family'] = "sans-serif"

# Metrics for Regression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# import Models and functions
from pages.linear_regression_model import lr_model
from pages.rf_regression_model import rf_model
from pages.load import load_data

# local path for storing data
file_path_transformed=r"C:\Users\jan.lade\OneDrive - Jedox AG\Documents\DHBW\6. Semester\Sales_Intelligence_Suite\data"


def ml_selection(data, include_target=True):
    """
    Selects target and model options.

    Parameters:
    - data (DataFrame): The Pandas DataFrame containing the dataset.

    Returns:
    - target_variable (str): The selected target variable.
    - model_selection (str): The selected model option (e.g., "Linear Regression", "Random Forest").
    - internal_features (DataFrame): The selected internal features.
    """
    if include_target:
        # Selectbox for target variable
        target_variable = st.selectbox("Select target measure", data.columns, index=len(data.columns) - 1)
    else:
        target_variable = data.columns[-1]  # Just pick the last column as the target variable
   
    # Select regression task
    model_selection = st.selectbox("Select a Model", ["Linear Regression", "Random Forest"], index=None)
    
    # Multiselect for internal features
    internal_features = data.drop(columns=[target_variable]) #st.multiselect("Select features", data.columns)

    if not model_selection:
        st.warning(":warning: Please select a task.")


    return target_variable, model_selection, internal_features


def x_and_y(data, internal_features, target_variable):
    """
    Creates X and y variables for modeling.

    Parameters:
    - data (DataFrame): The Pandas DataFrame containing the dataset.
    - internal_features (DataFrame): The selected internal features.
    - target_variable (str): The selected target variable.

    Returns:
    - X (DataFrame): The features used for modeling.
    - y (Series): The target variable used for modeling.
    """        
    X = internal_features
    y = data[target_variable]

    # Store X and y in session state
    # st.session_state.X = X
    # st.session_state.y = y

    return X, y


def compute_correlation(X, y, target_variable):
    """Function to compute correlation with a button for visualization"""

    # Button to initiate the correlation computation
    compute_correlation_button = st.button("Compute Correlations")

    plot_placeholder = st.empty()

    if compute_correlation_button:
        # Calculate correlation with the target variable
        correlations = X.corrwith(y)

        # Plot correlation with the target variable
        plt.figure(figsize=(12, 8), layout='constrained')
        sns.barplot(x=correlations, y=correlations.index)
        plt.xlabel('Correlation', fontsize=16)
        plt.ylabel('Features', fontsize=16)
        plt.title(f'Feature Correlations with target measure: {target_variable}', fontsize=26)
        plt.tight_layout()

        # Display the plot using Streamlit
        st.pyplot(plt.gcf())

        # Store the plot in the session state
        st.session_state.plot = plt.gcf()
    else:
        # If the button is not clicked, check if the plot is in the session state and display it
        if "plot" in st.session_state:
            plot_placeholder.pyplot(st.session_state.plot)


def train_model(X, y, model_selection, selected_file):
    """
    Trains the selected model.

    Parameters:
    - X (DataFrame): The features used for modeling.
    - y (Series): The target variable used for modeling.
    - model_selection (str): The selected model for training.

    Returns:
    - y_pred (array): Predicted values.
    - y_test (array): True target values from the test set.
    - x_pred (array): Predicted values from the training set.
    - y_train (array): True target values from the training set.
    - model_selection (str): The selected model for training.
    - best_params (dict): Best hyperparameters found by the model.
    """
    if st.button("Train Model", help="Results in training, may take some time!", key="trainmodelbutton"):
        if model_selection == "Linear Regression":
            y_pred, y_test, x_pred, y_train, best_params = lr_model(X, y, selected_file)
            st.success(f"{model_selection} trained successfully :clap:")
            
            # Store the results in session state
            st.session_state.y_test = y_test
            st.session_state.y_pred = y_pred
            st.session_state.y_train = y_train
            st.session_state.x_pred = x_pred
            st.session_state.model_selection = model_selection
            st.session_state.best_params = best_params

            return y_pred, y_test, x_pred, y_train, model_selection, best_params

        elif model_selection == "Random Forest":
            y_pred, y_test, x_pred, y_train, best_params = rf_model(X, y, selected_file)
            st.success(f"{model_selection} trained successfully! :clap:")

            # Store the results in session state
            st.session_state.y_test = y_test
            st.session_state.y_pred = y_pred
            st.session_state.y_train = y_train
            st.session_state.x_pred = x_pred
            st.session_state.model_selection = model_selection

            return y_pred, y_test, x_pred, y_train, model_selection, best_params


@st.cache_data
def get_metrics(y_pred, y_test, x_pred, y_train, model_selection):
    """
    Calculate and display evaluation metrics for the trained model.

    Parameters:
    - y_pred (array): Predicted values.
    - y_test (array): True target values from the test set.
    - x_pred (array): Predicted values from the training set.
    - y_train (array): True target values from the training set.
    - model_selection (str): The selected model for training.

    Returns:
    - metrics_table (DataFrame): A table containing evaluation metrics for the model.
    """        
    # Calculate Absolute Squared Error (MAE)
    mae_test = mean_absolute_error(y_test, y_pred)
    mae_train = mean_absolute_error(y_train, x_pred)

    # Calculate Mean Squared Error (MSE)
    mse_test = mean_squared_error(y_test, y_pred)
    mse_train = mean_squared_error(y_train, x_pred)

    # Calculate R-squared (R2)
    r2_test = r2_score(y_test, y_pred)
    r2_train = r2_score(y_train, x_pred)

    # Create a table with the metrics
    metrics_data = {
        "Dataset": ["Train Set", "Test Set"],
        "R-squared (R2)": [r2_train, r2_test],
        "Mean Absolute Error (MAE)": [mae_train, mae_test],
        "Mean Squared Error (MSE)": [mse_train, mse_test]
        }

    metrics_table = pd.DataFrame(metrics_data)

    # Display the table
    st.table(metrics_table)

    return metrics_table


@st.cache_data
def plot_results(y_pred, y_test, x_pred, y_train, model_selection):
    """
    Plot the results of a regression task by comparing actual vs. predicted values.

    Parameters:
    - y_pred (array): Predicted values.
    - y_test (array): True target values from the test set.
    - x_pred (array): Predicted values from the training set.
    - y_train (array): True target values from the training set.
    - model_selection (str): The selected model for training.

    Returns:
    - None
    """    
    # Function to plot actual vs. predicted values and display the plot
    plt.figure(figsize=(12, 8), layout='constrained')

    # Plot actual vs. predicted values
    plt.scatter(y_train, x_pred, alpha=0.7)

    # Add Ideal Prediction Line (First Bisector)
    min_val = min(np.min(y_train), np.min(x_pred))
    max_val = max(np.max(y_train), np.max(x_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Ideal Prediction Line')
        
    plt.xlabel('Actual Values', fontsize=16)
    plt.ylabel('Predicted Values', fontsize=16)
    plt.title('Actual vs. Predicted Values on Training Set', fontsize=26)
    plt.tight_layout()
    plt.legend()

    st.pyplot(plt.gcf())

    st.divider()

    plt.figure(figsize=(12, 8), layout='constrained')

    # Plot actual vs. predicted values
    plt.scatter(y_test, y_pred, alpha=0.7)

    # Add Ideal Prediction Line (First Bisector)
    min_val = min(np.min(y_test), np.min(y_pred))
    max_val = max(np.max(y_test), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Ideal Prediction Line')
        
    plt.xlabel('Actual Values', fontsize=16)
    plt.ylabel('Predicted Values', fontsize=16)
    plt.title('Actual vs. Predicted Values on Test Set', fontsize=26)
    plt.tight_layout()
    plt.legend()

    st.pyplot(plt.gcf())


def wizard():
    """
    Display the ML Wizard home screen, allowing users to select models, features, train models, and view results.

    Returns:
    - None
    """   
    st.header(":violet[ML] Wizard", divider="violet")
    # Step 1: Load transformed Data
    df = load_data(file_path_transformed)

    if df is not None:
        data, file_name = df
        st.write(data)
        st.divider()

        # Step 2: Feature & Target Selection
        st.subheader("Model Selection", divider="violet")
        target_variable, model_selection, internal_features = ml_selection(data)

        if model_selection is not None:
            # Step 4: Creating X and y
            X, y = x_and_y(data, internal_features, target_variable)
            st.divider()
            
            st.subheader("Correlation Analysis", divider="violet")
            compute_correlation(X, y, target_variable)
            st.divider()

            # Step 4: Train Model
            st.subheader("Training", divider="violet")
            result = train_model(X, y, model_selection, file_name) 

            if result is not None: # to handle event if button is not pressed
                # Unpack the result tuple
                y_pred, y_test, x_pred, y_train, model_selection, best_params = result
                
            else:
                st.warning("No model trained. Please click the train model button to train the model.")
                st.stop()

            
            # Create an expander to display best_params and metrics
            with st.expander("Model Details"):
                # Display best_params
                st.subheader("GridSearchCV Hyperparameters:", divider="violet")
                st.write(best_params)

                # Display metrics
                st.subheader("Metrics:", divider="violet")
                get_metrics(y_pred, y_test, x_pred, y_train, model_selection)

                # Display plots
                st.subheader("Plot Results:", divider="violet")
                plot_results(y_pred, y_test, x_pred, y_train, model_selection)



if __name__ == "__main__":
    st.page_link("app.py", label="Zurück zur Startseite", icon="🏠")
    wizard()