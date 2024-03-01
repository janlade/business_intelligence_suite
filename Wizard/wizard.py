import streamlit as st

from ETL import ETL

def wizard():
    st.header(":violet[W]izard", divider="violet")

if __name__ == "__main__":
    # st.page_link("ETL.py", label="Zurück zur Startseite", icon="🏠")
    wizard()