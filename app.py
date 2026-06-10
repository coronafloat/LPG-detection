import streamlit as st

from views.about import render_about
from views.contact import render_contact
from views.datasets import render_datasets
from views.home import render_home
from views.styles import inject_styles


st.set_page_config(page_title="LPG Gas Detection - YOLOv11", layout="wide")


def main():
    inject_styles()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Home", "About", "Datasets", "Contact"],
        label_visibility="collapsed",
    )

    if page == "Home":
        render_home()
    elif page == "About":
        render_about()
    elif page == "Datasets":
        render_datasets()
    elif page == "Contact":
        render_contact()


if __name__ == "__main__":
    main()
