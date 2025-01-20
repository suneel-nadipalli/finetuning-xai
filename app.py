import sys
import streamlit as st

sys.path.append('..')

from components.landing_page import landing_page
from components.viz_page import viz_page
from components.about_page import about_page

from streamlit_navigation_bar import st_navbar

st.set_page_config(page_title="Transformer Visualization App", page_icon="🔍", layout="wide")

# st.title('SAE: Sentiment Analysis Engine')

# Sidebar navigation
page = st_navbar(["Home", "Visualization", "About"])

# Render the selected page
if page == "Home":
    landing_page()

elif page == "Visualization":
    viz_page()

elif page == "About":
    about_page()
