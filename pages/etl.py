import streamlit as st

def etl():
    st.header(":violet[E]TL", divider="violet")
    st.write("Debus Project 1.0")
    st.page_link("pages/extract.py", label="Extract", icon= "🔷")
    st.page_link("pages/transform.py", label="Transform", icon= "🔶")
    st.page_link("pages/load.py", label="Load", icon= "🔷")

if __name__ == "__main__":
    st.page_link("app.py", label="Zurück zur Startseite", icon="🏠")
    etl()