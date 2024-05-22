import streamlit as st


horizontal_bar = "<hr style='margin-top: 0; margin-bottom: 0; height: 1px; border: 1px solid #749BC2;'><br>" 
horizontal_bar_big = "<hr style='margin-top: 0; margin-bottom: 0; height: 3px; border: 3px solid #749BC2;'><br>" 

def resouces_page ():

    st.markdown(horizontal_bar_big , True) 
    st.markdown("""

    #### Resources for Further Learning:

    For doctors interested in deepening their knowledge of Alzheimer's disease prediction using MRI scans, here are some valuable resources to explore:

    1. **Research Papers:**
    - Dive into peer-reviewed research articles and scientific papers covering various aspects of Alzheimer's disease diagnosis and prediction.
    - Access reputable databases like PubMed and Google Scholar, along with leading academic journals such as Neurology, Alzheimer's & Dementia, and the Journal of Alzheimer's Disease.

    2. **Clinical Guidelines:**
    - Review established clinical guidelines and protocols endorsed by reputable organizations in the field.
    - Explore comprehensive guidelines provided by esteemed institutions like the Alzheimer's Association, the National Institute on Aging (NIA), and the World Health Organization (WHO).

    3. **Educational Materials:**
    - Engage with educational resources such as online courses, webinars, and seminars tailored to healthcare professionals.
    - Platforms like Coursera, edX, and professional medical associations offer courses covering neuroimaging techniques, Alzheimer's disease pathology, and predictive modeling.

    4. **Professional Conferences:**
    - Participate in medical conferences, symposiums, and workshops dedicated to Alzheimer's disease research and clinical practice.
    - Gain valuable insights and networking opportunities at renowned events like the Alzheimer's Association International Conference (AAIC), the American Academy of Neurology (AAN) Annual Meeting, and the Radiological Society of North America (RSNA) Annual Meeting.

    5. **Online Forums and Discussion Groups:**
    - Join online forums, discussion groups, and social media communities focused on Alzheimer's disease research and clinical practice.
    - Connect with peers, share knowledge, and collaborate on advancements in the field through platforms like ResearchGate, Sermo, and specialized medical forums.

    6. **Books and Textbooks:**
    - Explore authoritative books and textbooks offering in-depth insights into neurology, neuroimaging, and Alzheimer's disease diagnosis.
    - Recommended reads include "Alzheimer's Disease: Diagnosis and Clinical Management" by Serge Gauthier and "Neuroimaging in Dementia" edited by Frederik Barkhof.

    By leveraging these diverse resources, doctors can deepen their understanding of Alzheimer's disease prediction models, stay abreast of the latest advancements, and contribute to enhancing patient care and outcomes.

    """ ,unsafe_allow_html=True)

    st.image("./project4/img/Alzheimer2.png")
