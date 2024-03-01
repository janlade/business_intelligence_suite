from pages.extract import *
from pages.transform import *
import streamlit as st

def etl():

    st.header(":violet[S]ales Intelligence Suite", divider="violet")
    st.write("Debus Project 1.0")
    st.page_link("pages/extract.py", label="Extract", icon= "🔷")
    st.page_link("pages/transform.py", label="Transform", icon= "🔶")
    st.page_link("pages/load.py", label="Load", icon= "🔷")


if __name__ == "__main__":
    etl()