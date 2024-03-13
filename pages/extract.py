import streamlit as st
import pandas as pd
import os

file_path_origin=r"C:\Users\jan.lade\OneDrive - Jedox AG\Documents\DHBW\6. Semester\Sales_Intelligence_Suite\data\original_data"


def load_data_widget():
    """Function for loading CSV Data"""

    # File uploader for CSV data
    uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # Display the data table
        data = pd.read_csv(uploaded_file)
        st.write(uploaded_file.name)
        st.dataframe(data)
        return data, uploaded_file.name
    
    else:
        st.info("Please upload a file.")
        return None, None


def save_data(data, file_path, file_name):
    """
    Save the provided Pandas DataFrame to a CSV file.

    Parameters:
    - data (DataFrame): The Pandas DataFrame to be saved.
    - file_path (str): The directory path where the data will be saved.
    - file_name (str): The name of the file.

    Returns:
    - None
    """
    try:
        if data is not None:
            # Construct the full file path with the original filename and extension
            full_file_path = os.path.join(file_path, f"{file_name}")
            # st.write(full_file_path)

            # if st.button("Save Data"):
            # Save the data to the constructed file path
            data.to_csv(full_file_path, index=False)

            st.success("Data saved successfully.")
        else:
            st.info("No data to save.")
    except Exception as e:
        st.error(f"Error saving data: {str(e)}")


def extract():
    """
    Extract data from the loaded source and save it to the specified file path.

    Steps:
    1. Load Data: Use the load_data_widget function to load the data.
    2. Save Data: Save the loaded data to the original file path.

    Returns:
    - None
    """    
    st.header(":violet[E]xtract", divider="violet")
    data, file_name = load_data_widget()

    if data is not None:
        save_data(data, file_path_origin, file_name)



if __name__ == "__main__":
    st.page_link("pages/etl.py", label="Zurück", icon="🏠")
    extract()