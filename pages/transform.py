__author__ = "Jan Lade"
__copyright__ = "Copyright 2024, Jan Lade"
__credits__ = ["Jan Lade", "Tom Debus"]
__version__ = "1.0"
__maintainer__ = "Jan Lade"
__status__ = "Production"


#imports
from pages.extract import save_data
from pages.load import load_data
from scipy.stats.mstats import winsorize
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# local path for storing data
file_path_transformed=r"C:\Users\jan.lade\OneDrive - Jedox AG\Documents\DHBW\6. Semester\Sales_Intelligence_Suite\data"
file_path_origin=r"C:\Users\jan.lade\OneDrive - Jedox AG\Documents\DHBW\6. Semester\Sales_Intelligence_Suite\data\original_data"


def data_selection(data):
    """
    Function to select date column, features, and target variable using Streamlit.

    Parameters:
    - data: Pandas DataFrame containing the dataset.

    Returns:
    - selected_features: List of selected features.
    """
    # Create a dictionary to store selected options
    selected_options = {}

    # Select column for indexing
    selected_options['index'] = st.selectbox("Select a column from the dataset to use as an index:", ["None"] + data.columns.tolist(), index=None)

    return selected_options


def eda(data):
    """Function for selecting EDA options"""

    # Create a selectbox for EDA options
    eda_option = st.selectbox("Select EDA Option", ["Data Shape", "Data Description", "Missing Values", "Data Types"], index=None)

    # Perform EDA based on the user's choice
    if eda_option == "Data Shape":
        # st.subheader("Data Shape")
        st.write("Number of Rows:", data.shape[0])
        st.write("Number of Columns:", data.shape[1])

    elif eda_option == "Data Description":
        # st.subheader("Data Description")
        st.write(data.describe())

    elif eda_option == "Missing Values":
        # st.subheader("Missing Values")
        missing_data = data.isnull().sum()
        st.write(missing_data)
    
    elif eda_option == "Data Types":
        # st.subheader("Data Types")
        data_types = data.dtypes
        st.write(data_types)


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


def index_data(data, selected_options):
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

    # Index the dataset by the selected index column
    data[index_column] = pd.to_datetime(data[index_column], format="mixed", dayfirst=True).dt.date

    # Set the index
    data.set_index(index_column, inplace=True)  
    # st.write("Index of the DataFrame:", data.index.name)

    if data.index.name:
        st.success("Selection confirmed")
        # st.success(f"Dataset successfully indexed by {index_column}")
    else:
        st.error("No index column is set.")

    return data


def encode_string_features(data):
    """
    Encode string features in the dataset using LabelEncoder.

    Parameters:
    - data (DataFrame): The Pandas DataFrame containing the dataset.

    Returns:
    - data (DataFrame): The updated Pandas DataFrame with encoded string features.
    """

    # Check if any selected feature is of string data type and encode it with LabelEncoder
    string_features = [col for col in data.columns if data[col].dtype == object]
    
    if any(data[col].dtype == object for col in data.columns):
        string_features_str = ", ".join(string_features)
        st.warning(f":warning: The selected features [{string_features_str}] are object values and will be encoded")

        label_classes = {} # store label for each class in a dict
        for col in string_features:
            label = LabelEncoder()
            label.fit(data[col].drop_duplicates())
            data[col] = label.transform(data[col])
    
            classes = label.classes_
            label_classes[col] = classes
    
    # st.write(label_classes)
    return data


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
    3. Automatically encodes object values with LabelEncoder.
    4. Handle Missing Values: Impute or delete missing values based on user selection.
    5. Handle Outliers: Apply outlier handling techniques such as Winsorization or Transformation.
    6. Save Data: Save the processed data to a CSV file.

    Returns:
    - None
    """
    st.header(":violet[T]ransform", divider="violet")
    
    # Step 1: Data Loading
    st.subheader("Data Transformation")
    df = load_data(file_path_origin)
    
    if df is not None:
        data, file_name = df
    
        st.write(data)
        st.divider()
        eda(data) # Options to view exploratory data analysis

        # Step 2: Data Selection    
        if data is not None:
            st.divider()
            selected_options = data_selection(data)
            # st.write("Selected Options:", selected_options)

            if st.button("Confirm Selection", key="confirm_selection_button"):
                if selected_options['index'] == 'None':
                    st.session_state.data_selected = data.copy()  # No index selected, so keep the original data
                    st.write(data)
                else:
                    data_selected = index_data(data, selected_options)
                    st.session_state.data_selected = data_selected  # Store the selected data in session state
                    st.write(data_selected)

            # Call encode_string_features function
            if "data_selected" in st.session_state:
                data_encoded = encode_string_features(st.session_state.data_selected)
                st.session_state.data_encoded = data_encoded
                # st.write("data_encoded", data_encoded)

            # Step 3: Handle Missing Values
            if "data_encoded" in st.session_state: # Check if data_selected is stored in session state
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
                    save_data(st.session_state.outlier_processed_data, file_path_transformed, file_name)

            

if __name__ == "__main__":
    st.page_link("pages/etl.py", label="Zurück", icon="🏠")
    transform()