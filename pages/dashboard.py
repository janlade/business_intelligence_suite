import streamlit as st
import joblib

from pages.load import load_data

file_path_transformed=r"C:\Users\jan.lade\OneDrive - Jedox AG\Documents\DHBW\6. Semester\Sales_Intelligence_Suite\data"



def load_model(model_option):
    model_filename = None

    if model_option == "Random Forest Regressor":
        model_filename = 'best_random_forest_regressor_model.pkl'
    elif model_option == "Regression Deep Learning":
        model_filename = 'best_regression_deep_learning_model.pkl'
    elif model_option == "Random Forest Classifier":
        model_filename = 'best_random_forest_classifier_model.pkl'
    elif model_option == "Classification Deep Learning":
        model_filename = 'best_classification_deep_learning_model.pkl'

    if model_filename is not None:
        model = joblib.load(model_filename)
    else:
        model = None

    return model

def make_predictions(internal_features, model_option, data, task_suggestion):
    st.sidebar.header("Make Predictions")
    user_input = {}

    if task_suggestion == "Classification":
        # Retrieve the scaler object from the session state
        if 'scaler' in st.session_state:
            scaler = st.session_state.scaler
            if internal_features is not None:
                data[internal_features] = scaler.inverse_transform(data[internal_features])
                
                for feature in internal_features:
                    # Calculate the min and max values based on the actual data
                    min_value = float(data[feature].min())
                    max_value = float(data[feature].max())
                    # Create sliders for each feature, set a reasonable range based on your data
                    user_input[feature] = st.sidebar.slider(f"Enter {feature}:", min_value=min_value, max_value=max_value, value=(min_value + max_value) / 2)
            # st.warning("Scaling has been reversed")
        else:
            st.warning("No scaler object found. Make sure the data was previously scaled.")

    if task_suggestion == "Regression":
        for feature in internal_features:
            # Calculate the min and max values based on the actual data
            min_value = float(data[feature].min())
            max_value = float(data[feature].max())
            # Create sliders for each feature, set a reasonable range based on your data
            user_input[feature] = st.sidebar.slider(f"Enter {feature}:", min_value=min_value, max_value=max_value, value=(min_value + max_value) / 2, step=(max_value - min_value) / 100)

    # Make predictions based on user input
    model = load_model(model_option)
    
    if model:
        try:
            user_input_data = pd.DataFrame([user_input])
            predictions = model.predict(user_input_data)

            # Display predictions to the user
            st.sidebar.subheader(f"Predictions with {model_option:}")
            st.sidebar.write("Predicted Value:", predictions[0])

        except ValueError:
            st.warning("An error occurred - The data does not match the data that was passed during the last fit. Please train the model again!")

    else:
        st.sidebar.error("No model loaded. Please train a model first.")

def dashboard():
    
    st.header(":violet[D]ashboard", divider="violet")
    
    # Step 1: Load transformed Data
    df = load_data(file_path_transformed)
    st.divider()

    if df is not None:
        data, file_name = df
        # st.write(data)

        st.divider()




if __name__ == "__main__":
    st.page_link("app.py", label="Zurück zur Startseite", icon="🏠")
    dashboard()