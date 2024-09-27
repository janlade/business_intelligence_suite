__author__ = "Jan Lade"
__copyright__ = "Copyright 2024, Jan Lade"
__credits__ = ["Jan Lade", "Tom Debus"]
__version__ = "1.0"
__maintainer__ = "Jan Lade"
__status__ = "Production"


#imports
import streamlit as st


def etl():
    """
    Displaying the ETL home screen.

    Contains:
    1. Extract: Extract data from the loaded source and save it to the specified file path (locally)
    2. Transform: Perform data transformation including data loading, selection, missing value handling, outlier handling, and data saving.
    3. Load: Load original and transformed data.

    Returns:
    - None
    """    
    st.header(":violet[E]TL", divider="violet")
    st.write("Debus Project 1.0")
    st.page_link("pages/extract.py", label="Extract", icon= "🔷")
    st.page_link("pages/transform.py", label="Transform", icon= "🔶")
    st.page_link("pages/load.py", label="Load", icon= "🔷")



if __name__ == "__main__":
    st.page_link("app.py", label="Landing Page", icon="🏠")
    etl()