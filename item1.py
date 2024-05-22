import hydralit_components as hc
import platform
import pandas as pd
import requests
import streamlit as st
# import streamlit_analytics

import streamlit_lottie
import time
import json

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

import os



st.set_page_config(
    page_title='brain MRI',
    page_icon="🧠",
    initial_sidebar_state="expanded"
)


###lottie###

# def load_lottiefile(filepath: str):
#     with open(filepath, "r") as f:
#         return json.load(f)


# if 'lottie' not in st.session_state:
#     st.session_state.lottie = False

# if not st.session_state.lottie: 
#     lottfinder = load_lottiefile("project4/.streamlit/TFinder_logo_animated.json")
#     st.lottie(lottfinder, speed=1.3, loop=False)
#     time.sleep(2)
#     st.session_state.lottie = True
#     rerun()    

###########

###style###

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

# st.sidebar.image("./project4/img/brain.png")

# # Help
# st.sidebar.title("Help")
# # with st.sidebar.expander("Video tutorials"):
# #     st.write('coming soon')

# with st.sidebar.expander("Regulatory regions extractor"):
#     st.subheader("Gene ID:")
#     st.write("ENTREZ_GENE_ID of NCBI and gene names are allowed.")
#     st.write(
#         "There is no limit to the number of gene names/ENTREZ_GENE_ID. Add them with a line break "
#         "(like those displayed by default). You can mix ENTREZ_GENE_ID and gene names as long as they "
#         "are of the same species.")
#     st.write("**Advance mode** allows you to select multiple species for genes")
#     st.write("⚠️A **Check genes avaibility** button allows you to analyse if your gene is accessible for species"
#              "and if ID is correct. Please use it. ")

#     st.subheader("Species:")
#     st.write("Human, mouse, rat, drosophila and zebrafish are allowed.")
#     st.write("If you use several ENTREZ_GENE_ID/gene names, make sure you select the correct species.")
#     st.write("⚠️Use **Check genes avaibility** button for checking species")
#     st.write("**Advance mode** allows you to select multiple species for genes")

#     st.subheader("Regulatory regions and Upstream/Downstream")
#     st.write("Distance to Transcription Start Site (TSS) or gene end in bp.")
#     st.image("./project4/img/brainposter.png")

#     st.subheader("Sequences to analyse:")
#     st.write(
#         'Use "Find promoter/extractor" button or paste your sequences. FASTA format allowed and required for multiple sequences.')
#     st.write(
#         'FASTA format: All sequences must have the TSS at the same distance, otherwise you assume the inconsistency of the positions of found sequences')

# with st.sidebar.expander("Individual Motif Finder"):
#     st.subheader("Responsive element:")
#     st.write('For **Individual Motif**: IUPAC code is authorized')
#     st.write(
#         'For **PWM**: You can generate a PWM with several sequences in FASTA format or use a PWM already generated with our tools  (same length required)')
#     st.write("For **JASPAR_ID** option, use the JASPAR_ID of your transcription factor.")
#     st.image("./project4/img/brain.png")
#     st.subheader("Transcription Start Site (TSS) or gene end:")
#     st.write('Distance to Transcription Start Site (TSS) or gene end in bp')
#     st.write('Note: If you use Step 1 , it will be defined automatically.')
#     st.subheader("Relative Score Threshold:")
#     st.write('Eliminates responsive element with Relative Score < threshold')
#     st.write(
#         'The Relative Score represents the Score calculated for each k-mer of the length of the PWM in the given sequence where each corresponding probability is added according to each nucleotide. This Score is then normalized to the maximum and minimum PWM Score.')
#     st.subheader('_p-value_')
#     st.write(
#         'The p-value calculation takes time so it is optional. it represents the probability that a random generated sequence of the lenght of the PWM with the nucleotide proportions of the sequence has a score greater than or equal to the element found.')

# st.sidebar.title("Servers status",
#                  help='✅: servers are reachable. You can use extract regions via NCBI/use the JASPAR_IDs\n\n❌: servers are unreachable. You can still use TFinder if you have a sequence in FASTA format and a pattern to search in the sequence')

# if st.sidebar.button("Check"):
#     with st.sidebar:
#         with st.spinner('Please wait...'):
#             response = requests.get(
#                 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=gene&term=nos2[Gene%20Name]+AND+human[Organism]&retmode=json&rettype=xml')
#             response1 = requests.get('https://jaspar.elixir.no/api/v1/matrix/MA0106.1')

#             ncbi_status = "✅" if response.status_code == 200 else "❌"
#             jaspar_status = "✅" if response1.status_code == 200 else "❌"

#             st.session_state['ncbi_status'] = ncbi_status
#             st.session_state['jaspar_status'] = jaspar_status

#             data = {
#                 "NCBI": [ncbi_status],
#                 "JASPAR": [jaspar_status]
#             }

#             df = pd.DataFrame(data, index=["Servers status"])

#             st.sidebar.table(df)

# st.sidebar.title("More")
# st.sidebar.markdown(
#     "[Report a bug 🐞](https://github.com/Jumitti/TFinder/issues/new?assignees=&labels=bug&projects=&template=bug_report.md&title=%5BBUG%5D)")
# st.sidebar.markdown(
#     "[Need HELP 🆘](https://github.com/Jumitti/TFinder/issues/new?assignees=&labels=help+wanted&projects=&template=help.md&title=%5BHELP%5D)")
# st.sidebar.markdown(
#     "[Have a question 🤔](https://github.com/Jumitti/TFinder/issues/new?assignees=&labels=question&projects=&template=question_report.md&title=%5BQUESTION%5D)")
# st.sidebar.markdown(
#     "[Features request 💡](https://github.com/Jumitti/TFinder/issues/new?assignees=&labels=enhancement&projects=&template=feature_request.md&title=%5BFEATURE%5D)")
# st.sidebar.markdown("[Want to talk ? 🙋🏼‍♂](https://github.com/Jumitti/TFinder/discussions)")

# # streamlit_analytics.stop_tracking()
# # views = streamlit_analytics.main.counts["total_pageviews"]
# try:
#     local_test = platform.processor()
#     if local_test == "":
#         unique_users = st.secrets['unique_users']
#         st.sidebar.markdown(f"Unique users 👥: {unique_users}")
#         st.session_state["LOCAL"] = 'False'
# except Exception as e:
#     st.session_state["LOCAL"] = 'True'
#     st.sidebar.markdown(f"TFinder Local Version")



#############

# Define the content of the sidebar
import streamlit as st

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

st.sidebar.image("./project4/img/Alzheimer4.png", use_column_width=True)

























