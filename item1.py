import hydralit_components as hc
import pandas as pd
import streamlit as st

import os

from navigation.doctor import doctor_page
from navigation.more import more_page
from navigation.home import home_page
from navigation.patient import patient_page

from utils.components import footer_style, footer
try:
    from streamlit import rerun as rerun
except ImportError:
    # conditional import for streamlit version <1.27
    from streamlit import experimental_rerun as rerun


# Determine the root directory of the project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


st.set_page_config(
    page_title='brain MRI',
    page_icon="🧠",
    initial_sidebar_state="expanded"
)


max_width_str = f"max-width: {75}%;"

st.markdown(f"""
        <style>
        .appview-container .main .block-container{{{max_width_str}}}
        </style>
        """,
            unsafe_allow_html=True,
            )

st.markdown("""
        <style>
               .block-container {
                    padding-top: 0rem;
                    padding-bottom: 0rem;
                    
                }
        </style>
        """, unsafe_allow_html=True)

# Footer

st.markdown(footer_style, unsafe_allow_html=True)

###########

###NavBar###

HOME = 'Home'
PATIENT = 'Patient'
DOCTOR = 'Doctor'
MORE = 'More'

tabs = [
    HOME,
    PATIENT,
    DOCTOR,
    MORE,
]

option_data = [
    {'icon': "🏠", 'label': HOME},
    {'icon': "👎", 'label': PATIENT},
    {'icon': "👨‍⚕️", 'label': DOCTOR},
    {'icon': "✉️", 'label': MORE},
]

over_theme = {'txc_inactive': 'black', 'menu_background': '#D6E5FA', 'txc_active': 'white', 'option_active': '#749BC2'}
font_fmt = {'font-class': 'h3', 'font-size': '50%'}

chosen_tab = hc.option_bar(
    option_definition=option_data,
    title='',
    key='PrimaryOptionx',
    override_theme=over_theme,
    horizontal_orientation=True)


if chosen_tab == HOME:
    home_page()

elif chosen_tab == PATIENT:
    patient_page()

elif chosen_tab == DOCTOR:
    doctor_page()

elif chosen_tab == MORE:
    more_page()

###########


###end###
for i in range(4):
    st.markdown('#')
st.markdown(footer, unsafe_allow_html=True)   # from utils components

#########


###sidebar###


# Define the content of the sidebar
st.sidebar.title('Alzheimer\'s Disease Prediction')
st.sidebar.markdown('Welcome to the Alzheimer\'s Disease Prediction app!')

st.sidebar.subheader('About')
st.sidebar.info('This app uses deep learning models to predict the likelihood of Alzheimer\'s disease based on MRI brain scans.')

st.sidebar.subheader('Instructions')
st.sidebar.write('1. Upload an MRI brain image using the file uploader on the main page.')
st.sidebar.write('2. The app will analyze the image and provide predictions along with interpretations.')
st.sidebar.write('3. You can learn more about the prediction process and interpretation in the main section.')

st.sidebar.subheader('Resources')
st.sidebar.write('Explore further learning resources:')
st.sidebar.markdown('- [Research Papers](#)')
st.sidebar.markdown('- [Clinical Guidelines](#)')
st.sidebar.markdown('- [Educational Materials](#)')
st.sidebar.markdown('- [Professional Conferences](#)')
st.sidebar.markdown('- [Online Forums](#)')
st.sidebar.markdown('- [Books and Textbooks](#)')

st.sidebar.subheader('Contact Us')
st.sidebar.write('For inquiries or support, please contact:')
st.sidebar.markdown('- Email: contact@alzheimerprediction.com')
st.sidebar.markdown('- Phone: +1 (123) 456-7890')

st.sidebar.subheader('Disclaimer 😂')
st.sidebar.write('This app is for educational purposes only. Consult a healthcare professional for medical advice.')


# Adjust the path to the sidebar image
sidebar_image_path = os.path.join(ROOT_DIR, "img", "Alzheimer4.png")
st.sidebar.image(sidebar_image_path, use_column_width=True)

#############






