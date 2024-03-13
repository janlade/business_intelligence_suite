__author__ = "Jan Lade"
__copyright__ = "Copyright 2024, Jan Lade"
__credits__ = ["Jan Lade", "Tom Debus"]
__version__ = "1.0"
__maintainer__ = "Jan Lade"
__status__ = "Production"


#imports
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
    # st.page_link("pages/wizard.py", label="Transform", icon= "🔶")
    # st.page_link("pages/dashboard.py", label="Load", icon= "🔷")