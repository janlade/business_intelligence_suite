from pages import extract
from pages import load

import pandas as pd
import streamlit as st
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

file_path_transformed=r"C:\Users\jan.lade\OneDrive - Jedox AG\Documents\DHBW\6. Semester\Sales_Intelligence_Suite\data"
file_path_origin=r"C:\Users\jan.lade\OneDrive - Jedox AG\Documents\DHBW\6. Semester\Sales_Intelligence_Suite\data\original_data"


def data_selection(data):
    """
    Function to select date column, features, and target variable using Streamlit.

    Parameters:
    - data: Pandas DataFrame containing the dataset.

    Returns:
    - selected_features: List of selected features.
    - target_variable: Name of the selected target variable.
    """
    # Create a dictionary to store selected options
    selected_options = {}

    # Select column for indexing
    selected_options['index'] = st.selectbox("Select a column from the dataset to use as an index:", ["None"] + data.columns.tolist(), index=None)

    # Select target variable
    # selected_options['target_variable'] = st.selectbox("Select the target variable from the dataset:", data.columns, index=len(data.columns) - 1)


    return selected_options


def missing_value_handling():
    """
    Function to select the method for handling missing values.

    Returns:
    - selected_method: Method selected for handling missing values.
    """
    selected_method = st.selectbox("Select method for handling missing values:", ["None", "Imputation", "Deletion"], index=None)
    return selected_method


def outlier_handling():
    """
    Function to select the method for handling outliers.

    Returns:
    - selected_method: Method selected for handling outliers.
    """
    selected_method = st.selectbox("Select method for handling outliers:", ["None", "Winsorization", "Transformation"], index=None)
    return selected_method


def process_selected_options(data, selected_options):
    """
    Process the selected options.

    Parameters:
    - data: Pandas DataFrame containing the dataset.
    - selected_options: Dictionary containing selected options.

    Returns:
    - target_variable:
    - features:
    """
    index_column = selected_options['index']
    # target_variable_name = selected_options['target_variable']

    # Index the dataset by the selected index column
    data[index_column] = pd.to_datetime(data[index_column], format="mixed", dayfirst=True).dt.date
    # Set the index
    data.set_index(index_column, inplace=True)  
    # st.write("Index of the DataFrame:", data.index.name)

    # Select the target variable
    # target_variable = data[target_variable_name]

    # Create feature dataset by dropping the target variable
    # features = data.drop(target_variable_name, axis=1)

    if data.index.name:
        st.success("Selection Confirmed")
        # st.success(f"Dataset successfully indexed by {index_column}")
    else:
        st.error("No index column is set.")
    # Process further actions with selected options
    # st.write("Selected Index Column:", index_column)
    # st.write("Selected Target Variable:", target_variable_name)
    # st.write("Selected Features:", features.columns.tolist())
    
    return data#, target_variable, features


def impute_missing_values(data):
    """
    Handle missing values in a Pandas DataFrame using imputation and drops duplicates.

    Parameters:
    - data: Pandas DataFrame containing the dataset.

    Returns:
    - data_imputed: Pandas DataFrame with missing values imputed and rounded to zero decimal points.
    """
    # Identify numerical and categorical columns
    numerical_cols = data.select_dtypes(include=['number']).columns
    categorical_cols = data.select_dtypes(include=['object']).columns

    # st.write(numerical_cols)
    # st.write(categorical_cols)

    # Impute missing values for numerical columns with the mean
    for col in numerical_cols:
        if data[col].isnull().any():
            mean_value = data[col].mean()
            data[col].fillna(mean_value, inplace=True)
            data[col] = data[col].round(0)

    # Impute missing values for categorical columns with the mode
    for col in categorical_cols:
        if data[col].isnull().any():
            mode_value = data[col].mode().iloc[0]
            data[col].fillna(mode_value, inplace=True)
    
    # Remove duplicate rows
    data.drop_duplicates(inplace=True)

    st.success("Imputed Missing Values and deleted duplicates")
    return data


def winzorize_outliers(data):
    """
    Handle outliers in numerical columns of a Pandas DataFrame using winsorization.
    Winsorization retains the general distribution of the data while reducing the impact of extreme values.
    Meaing replacing extreme values (outliers) with less extreme values at a certain percentile, such as the 95th or 99th percentile.
    
    Parameters:
    - data: Pandas DataFrame containing the dataset with numerical columns.

    Returns:
    - data_handled: Pandas DataFrame with outliers handled using winsorization.
    """
    # Identify numerical columns
    numerical_cols = data.select_dtypes(include=['number']).columns

    # Winsorize numerical columns
    for col in numerical_cols:
        # Winsorize at the 5th and 95th percentiles
        data[col] = winsorize(data[col], limits=[0.05, 0.05])

    st.success("Outliers handled with winzorization")
    return data


def transform():
    """
    Perform data transformation including data loading, selection, missing value handling, outlier handling, and data saving.

    Steps:
    1. Data Loading: Load data from the specified file path.
    2. Data Selection: Allow the user to select a column for indexing.
    3. Handle Missing Values: Impute or delete missing values based on user selection.
    4. Handle Outliers: Apply outlier handling techniques such as Winsorization or Transformation.
    5. Save Data: Save the processed data to a CSV file.

    Returns:
    - None
    """
    st.header(":violet[T]ransform", divider="violet")
    # Step 1: Data Loading
    st.subheader("Data Transformation")
    df = load.load_data(file_path_origin)
    
    if df is not None:
        data, file_name = df
    
        st.write(data)

        # Step 2: Data Selection    
        if data is not None:
            st.divider()
            selected_options = data_selection(data)
            st.write("Selected Options:", selected_options)

            if st.button("Confirm Selection", key="confirm_selection_button"):
                if selected_options['index'] == 'None':
                    st.session_state.data_selected = data.copy()  # No index selected, so keep the original data
                    st.write(data)
                else:
                    data_selected = process_selected_options(data, selected_options)
                    st.write(data_selected)
                    st.session_state.data_selected = data_selected  # Store the selected data in session state


            # Step 3: Handle Missing Values
            if "data_selected" in st.session_state: # Check if data_selected is stored in session state
                st.divider()
                missing_value_method = missing_value_handling()

                # If "None" is selected, keep the data as is
                if missing_value_method is not None:
                    if missing_value_method == "None":  
                        if st.button("Confirm Selection", key="confirm_imputation_button"):
                            processed_data = st.session_state.data_selected
                            st.write(processed_data)
                            st.session_state.processed_data = processed_data

                    # If imputation is selected
                    elif missing_value_method == "Imputation":  
                        if st.button("Confirm Selection", key="confirm_imputation_button"):
                            processed_data  = impute_missing_values(st.session_state.data_selected)
                            st.write(processed_data )
                            st.session_state.processed_data = processed_data

                    # If deletion is selected
                    # elif missing_value_method == "Deletion":  
                    #     if st.button("Confirm Selection", key="confirm_deletion_button"):
                    #         processed_data = delete_missing_values(st.session_state.data_selected)
                    #         st.write(processed_data)
                    #         st.session_state.processed_data = processed_data


            # Step 4: Handle Outliers
            if "processed_data" in st.session_state: # Check if data_selected is stored in session state
                st.divider()
                outlier_method = outlier_handling()

                # If "None" is selected, keep the data as is
                if outlier_method is not None:
                    if outlier_method == "None":  
                        if st.button("Confirm Selection", key="confirm_outlier_handling_button"):
                            outlier_processed_data  = st.session_state.processed_data
                            st.session_state.outlier_processed_data = outlier_processed_data 
                            st.write(outlier_processed_data )

                    # If Winsorization is selected
                    elif outlier_method == "Winsorization":  
                        if st.button("Confirm Selection", key="confirm_outlier_handling_button"):
                            outlier_processed_data  = winzorize_outliers(st.session_state.processed_data)
                            st.write(outlier_processed_data)
                            st.session_state.outlier_processed_data = outlier_processed_data 

                    # If Transformation is selected
                    # elif outlier_method == "Transformation":  
                    #     if st.button("Confirm Selection", key="confirm_outlier_handling_button"):
                    #         outlier_processed_data  = transform_outliers(st.session_state.processed_data)
                    #         st.write(outlier_processed_data )
                    #         st.session_state.outlier_processed_data  = outlier_processed_data 


            # Last Step: Save Data
            if "outlier_processed_data" in st.session_state: # Check if outlier_processed_data is stored in session state
                st.divider()
                if st.button("Save Data", key="save_data_button"):
                    extract.save_data(st.session_state.outlier_processed_data, file_path_transformed, file_name)

            

if __name__ == "__main__":
    st.page_link("ETL.py", label="Zurück zur Startseite", icon="🏠")
    transform()