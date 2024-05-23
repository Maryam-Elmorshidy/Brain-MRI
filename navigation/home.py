import streamlit as st
import random
from PIL import Image
import os 
def home_page():
    horizontal_bar = "<hr style='margin-top: 0; margin-bottom: 0; height: 1px; border: 1px solid #749BC2;'><br>" 

    vDrive = os.path.splitdrive(os.getcwd())[0]
    vpth = "D:/4AI term 2/project graduation/streamlit/" if vDrive == "C:" else "./"

    #st.text("home_page")
    col1, col2 = st.columns(2)

    # Load a random image
    random.seed()
    image_path = vpth + random.choice(["./img/Alzheimer.jpg","./img/Alzheimer1.jpg","./img/brain1.jpg"])
    image = Image.open(image_path)

    # Display the heading in the first column
    col1.markdown("<strong style='font-weight: 10000'><span style='font-size: 150%'>Alzheimer's disease (AD)</span></strong> is a condition affecting the brain, where mental functions are destroyed. Over a period of time, it can result in irreversible loss of memory. Dr Alois Alzheimer identified the disease in the year 1906, after examining a dead woman’s brain for symptoms of memory loss. Alzheimer found abnormal clumps, called amyloid plaques, and tangled bundles, called neurofibrillary tangles, in her brain. Experts have it that around 5 million Americans older than 65 years of age are vulnerable to Alzheimer’s disease. The disease is more prevalent during an advanced age, usually over the age of 65 years. Every five years, the number of people with Alzheimer’s disease is seen to double.", unsafe_allow_html=True)

    #st.markdown(horizontal_bar, True)

    # Display the image in the second column with automatic column width
    col2.image(image, use_column_width='auto')

    # thin divider line
    
    st.markdown(horizontal_bar, True)   

    st.subheader("some of types Dementia :")
    st.markdown("""
                 - #### Mild Dementia

Signs and symptoms of mild dementia include memory loss, confusion about the location of familiar places, taking longer than usual to accomplish normal daily tasks, trouble handling money and paying bills, poor judgment leading to bad decisions, loss of spontaneity and sense of initiative, mood and personality changes, and increased anxiety or aggression.

- #### Moderate Dementia

Signs and symptoms include increased memory loss and confusion, shortened attention span, inappropriate angry outbursts, problem recognizing family and close friends, difficulty with language (reading, writing, numbers), inability to learn new things or cope with unexpected situations, difficulty organizing thoughts and thinking logically, repetitive statements or movements, occasional muscle twitches, restlessness, agitation, anxiety, tearfulness, wandering (especially in late afternoon or at night), hallucinations, delusions, suspiciousness, paranoia, irritability, loss of impulse control, inability to carry out activities that involve multiple steps in sequence (getting dressed, making coffee, setting the table).
- #### Severe Dementia

Signs and symptoms include weight loss, seizures, skin infections, difficulty swallowing, increased sleep, groaning, moaning or grunting, lack of bladder or bowel control.

If you feel that your condition is not related to outside causes, or that you may fall under one of the above categories, you should consider making an appointment with a physician or other medical specialist.
                
                """ , unsafe_allow_html=False)
    
    st.markdown(horizontal_bar, True)   
    st.markdown(horizontal_bar, True)   
    st.markdown("#### some image MRI :" , unsafe_allow_html=False)
    st.markdown("##### 1) MILD :" , unsafe_allow_html=False)

    c1, c2 , c3 , c4= st.columns(4)

    
    image_path = vpth + "./img_brain/mild/mild1.jpg"
    image1 = Image.open(image_path)

    c1.image(image1, use_column_width='auto')

    
    image_path = vpth + "./img_brain/mild/mild2.jpg"
    image2 = Image.open(image_path)

    c2.image(image2, use_column_width='auto')

    image_path = vpth + "./img_brain/mild/mild3.jpg"
    image3 = Image.open(image_path)

    c3.image(image3, use_column_width='auto')

    image_path = vpth + "./img_brain/mild/mild4.jpg"
    image4 = Image.open(image_path)

    c4.image(image4, use_column_width='auto')

    st.markdown(horizontal_bar, True)   

    st.markdown("##### 2) MODERATE :" , unsafe_allow_html=False)

    c1, c2 , c3 , c4= st.columns(4)

    image_path = vpth + "./img_brain/moderate/moderate1.jpg"
    image1 = Image.open(image_path)

    c1.image(image1, use_column_width='auto')

    
    image_path = vpth + "./img_brain/moderate/moderate2.jpg"
    image2 = Image.open(image_path)

    c2.image(image2, use_column_width='auto')

    image_path = vpth + "./img_brain/moderate/moderate3.jpg"
    image3 = Image.open(image_path)

    c3.image(image3, use_column_width='auto')

    image_path = vpth + "./img_brain/moderate/moderate4.jpg"
    image4 = Image.open(image_path)

    c4.image(image4, use_column_width='auto')

    st.markdown(horizontal_bar, True) 




#
