import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from pages.load import load_data

# Metrics for Regression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# import Models
from pages.linear_regression_model import lr_build
from pages.rf_regression_model import rf_build

file_path_transformed=r"C:\Users\jan.lade\OneDrive - Jedox AG\Documents\DHBW\6. Semester\Sales_Intelligence_Suite\data"


def ml_selection(data):
    """Function for feature and target selection"""

    # Selectbox for target variable
    target_variable = st.selectbox("Select target measure", data.columns, index=len(data.columns) - 1)

    # Select regression or classification task
    model_selection = st.selectbox("Select the task", ["Linear Regression", "Random Forest"], index=None)
    
    # Multiselect for internal features
    internal_features = data.drop(columns=[target_variable]) #st.multiselect("Select features", data.columns)

    # Check if any selected feature is of string data type and encode it with LabelEncoder
    string_features = [feature for feature in internal_features if data[feature].dtype == object]
    if any(data[feature].dtype == object for feature in internal_features) or data[target_variable].dtype == object:
        string_features_str = ", ".join(string_features)
        st.warning(f":warning: The selected features [{string_features_str}] are object values and will be encoded")

        for feature in string_features:
            label = LabelEncoder()
            label.fit(data[feature].drop_duplicates())
            data[feature] = label.transform(data[feature])

    if not model_selection:
        st.warning(":warning: Please select a task.")

    # # Check if target measure is only once selected and not in internal features aswell
    # if target_variable in internal_features:
    #     st.warning(":warning: Target measure can not be selected as a feature.")

    # if not internal_features:
    #     internal_features = None
    #     st.warning(":warning: Please select at least one feature.")

    # # Scale internal features if the task is "Classification"
    # if model_selection == "Classification":
    #     if internal_features is not None:
    #         scaler = StandardScaler()
    #         data[internal_features] = scaler.fit_transform(data[internal_features])
    #         st.warning("All selected features have been scaled")
    #         st.session_state.scaler = scaler

    st.write(target_variable)
    st.write(model_selection)
    st.write(internal_features)

    return target_variable, model_selection, internal_features


def x_and_y(data, internal_features, target_variable):
    """ Function to create X and y variable """
        
    
    X = internal_features
    y = data[target_variable]

    # st.write(X)
    # st.write(y)
    # st.success("Selection Confirmed!")

    # Store X and y in session state
    # st.session_state.X = X
    # st.session_state.y = y

    return X, y

def train_model(X, y, model_selection):
    """Function for model training and evaluation"""

    if st.button("Train Model", help="Results in training, may take some time!"):
        if model_selection == "Linear Regression":
            y_pred, y_test, x_pred, y_train, best_params = lr_build(X, y)
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
            y_pred, y_test, x_pred, y_train, best_params = rf_build(X, y)
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
    """Function to calculate and display evaluation metrics"""
        
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

def wizard():
    """
    Displaying the Wizard home screen.

    Contains:


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
        st.subheader("Model Selection")
        target_variable, model_selection, internal_features = ml_selection(data)

        if model_selection is not None:
            # Step 4: Creating X and y
            X, y = x_and_y(data, internal_features, target_variable)
            st.write("X", X)
            st.write("y", y)
            st.divider()

            # Step 4: Train Model
            y_pred, y_test, x_pred, y_train, model_selection, best_params = train_model(X, y, model_selection)
            st.write("GridSearchCV Hyperparameters:", best_params)

            # Step 5: Get Metrics
            get_metrics(y_pred, y_test, x_pred, y_train, model_selection)

if __name__ == "__main__":
    st.page_link("app.py", label="Zurück zur Startseite", icon="🏠")
    wizard()