import streamlit as st
import pandas as pd
import os

file_path_transformed=r"C:\Users\jan.lade\OneDrive - Jedox AG\Documents\DHBW\6. Semester\Sales_Intelligence_Suite\data"
file_path_origin=r"C:\Users\jan.lade\OneDrive - Jedox AG\Documents\DHBW\6. Semester\Sales_Intelligence_Suite\data\original_data"

def load_data(file_path):
    """
    Load data from the specified file path and return it as a Pandas DataFrame.

    Parameters:
    - file_path (str): The path to the directory containing the CSV files.

    Returns:
    - data (DataFrame): The loaded data as a Pandas DataFrame.
    - selected_file (str): The name of the selected CSV file.

    """
    try:
        # Get a list of CSV files in the specified directory
        csv_files = [f for f in os.listdir(file_path) if f.endswith('.csv')]
        
        # Let the user choose which CSV file to load
        if csv_files:
            selected_file = st.selectbox("Select a file to load:", csv_files, index=None, key="file_selector_"+file_path)
            if selected_file is not None:
                selected_file_path = os.path.join(file_path, selected_file)
                
                # Read the selected CSV file into a DataFrame
                data = pd.read_csv(selected_file_path)
                # st.write("Selected File:", selected_file)
                # st.write("Loaded Data:", data)
                st.success("Data loaded successfully.")
                return data, selected_file
            
            else:
                return None
        else:
            st.error("No CSV files found in the specified directory.")
            return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def load():
    """
    Load original and transformed data.

    Steps:
    1. Load Original Data: Load the original data from the specified file path.
    2. Load Transformed Data: Load the transformed data from the specified file path.

    Returns:
    - None
    """    
    st.header(":violet[L]oad", divider="violet")
    st.subheader("Load Original Data")
    df_original = load_data(file_path_origin)
    if df_original is not None:
        data_original, selected_file  = df_original
        st.write(data_original)

    st.divider()

    st.subheader("Load Transformed Data")
    df_transformed = load_data(file_path_transformed)
    if df_transformed is not None:
        data_transformed, selected_file = df_transformed
        st.write(data_transformed)



if __name__ == "__main__":
    st.page_link("pages/etl.py", label="Zurück", icon="🏠")
    load()