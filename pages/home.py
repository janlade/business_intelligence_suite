import streamlit as st

def home():
    """
    Displaying the starting home screen.

    Contains:
    1. ETL: Extract, Transform and Load Datasets.
    2. Wizard: Guide the user through setting up a Machine Learning Model
    3. Dashboard: Interaction with processed Data from ETL and Wizard

    Returns:
    - None
    """    
    st.header(":violet[B]usiness Intelligence Suite", divider="violet")
    st.write("Debus Project 1.0")
    st.page_link("pages/etl.py", label="ETL", icon= "🔷")
    st.page_link("pages/wizard.py", label="ML Wizard", icon= "🔶")
    # st.page_link("pages/dashboard.py", label="Load", icon= "🔷")