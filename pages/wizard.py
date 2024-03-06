import streamlit as st
from sklearn.preprocessing import LabelEncoder

from pages.load import load_data

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

    # st.write(target_variable)
    # st.write(model_selection)
    # st.write(internal_features)

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
            st.divider()

if __name__ == "__main__":
    st.page_link("app.py", label="Zurück zur Startseite", icon="🏠")
    wizard()