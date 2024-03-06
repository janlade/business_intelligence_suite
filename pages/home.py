import streamlit as st

def home():
    st.header(":violet[S]ales Intelligence Suite", divider="violet")
    st.write("Debus Project 1.0")
    st.page_link("pages/etl.py", label="ETL", icon= "🔷")
    # st.page_link("pages/wizard.py", label="Transform", icon= "🔶")
    # st.page_link("pages/dashboard.py", label="Load", icon= "🔷")