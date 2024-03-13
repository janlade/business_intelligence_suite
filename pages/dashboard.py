__author__ = "Jan Lade"
__copyright__ = "Copyright 2024, Jan Lade"
__credits__ = ["Jan Lade", "Tom Debus"]
__version__ = "1.0"
__maintainer__ = "Jan Lade"
__status__ = "Production"


#imports
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import functions
from pages.load import load_data
from pages.wizard import ml_selection

# local path for storing data
file_path_transformed=r"C:\Users\jan.lade\OneDrive - Jedox AG\Documents\DHBW\6. Semester\Sales_Intelligence_Suite\data"



def load_model(model_option, selected_file):
    model_filename = None

    if model_option == "Linear Regression":
        model_filename = f'{selected_file}_best_linear_regression_model.pkl'
    elif model_option == "Random Forest":
        model_filename = f'{selected_file}_best_random_forest_regressor_model.pkl'


    if model_filename is not None:
        model = joblib.load(model_filename)
    else:
        model = None

    return model, selected_file


def make_predictions(internal_features, model_option, data, selected_file):
    user_input = {}

    # Unpack the tuple returned by load_model
    model, selected_file = load_model(model_option, selected_file)

    for feature in internal_features:
        # Calculate the min and max values based on the actual data
        min_value = float(data[feature].min())
        max_value = float(data[feature].max())
        # Create sliders for each feature, set a reasonable range based on the data
        user_input[feature] = st.slider(f"Enter {feature}:", min_value=min_value, max_value=max_value, value=(min_value + max_value) / 2, step=(max_value - min_value) / 100)

    if model:
        try:
            user_input_data = pd.DataFrame([user_input])
            predictions = model.predict(user_input_data)

            # Display predictions to the user
            st.write("Predicted Value:", predictions[0])

        except ValueError:
            st.warning("An error occurred - The data does not match the data that was passed during the last fit. Please train the model again!")
    
    else:
        st.error("No model loaded. Please load a model first.")


def dashboard():
    
    st.header(":violet[D]ashboard", divider="violet")
    
    # Step 1: Load transformed Data
    df = load_data(file_path_transformed)
    st.divider()

    if df is not None:
        data, selected_file = df

        # Load needed input for predictions
        target_variable, model_selection, internal_features = ml_selection(data, include_target=False)
        
        # Make predictions
        make_predictions(internal_features, model_selection, data, selected_file)
        st.divider()



if __name__ == "__main__":
    st.page_link("app.py", label="Zurück zur Startseite", icon="🏠")
    dashboard()