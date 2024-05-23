import streamlit as st

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


from torchvision.models import resnet50, ResNet50_Weights

from captum.attr import IntegratedGradients
from captum.attr import visualization as viz

preprocess_func = ResNet50_Weights.IMAGENET1K_V2.transforms()
categories = np.array(ResNet50_Weights.IMAGENET1K_V2.meta["categories"])

@st.cache_resource
def load_model():
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.eval();
    return model

def make_prediction(model, processed_img):
    probs = model(processed_img.unsqueeze(0))
    probs = probs.softmax(1)
    probs = probs[0].detach().numpy()

    prob, idxs = probs[probs.argsort()[-5:][::-1]], probs.argsort()[-5:][::-1]
    return prob, idxs

def interpret_prediction(model, processed_img, target):
    interpretation_algo = IntegratedGradients(model)
    feature_imp = interpretation_algo.attribute(processed_img.unsqueeze(0), target=int(target))
    feature_imp = feature_imp[0].numpy()
    feature_imp = feature_imp.transpose(1,2,0)

    return feature_imp




def patient_page():
    horizontal_bar = "<hr style='margin-top: 0; margin-bottom: 0; height: 1px; border: 1px solid #749BC2;'><br>" 
    c1,c2 = st.columns(2)

    c1.markdown("""
### Introduction:

Alzheimer’s disease (AD) is the most common neurological disease due to a disorder known as cognitive impairment, which progresses to deterioration in cognitive abilities, behavioral changes, and memory loss; AD affects the adaptability that needs to be promoted. Despite the scientific advancement in the medical field, there is no active cure for AD, but the effective method for AD is to slow its progression. Therefore, early detection of Alzheimer’s symptoms in its first stage is vital to prevent its progression to advanced stages. Dementia is one of the most common forms of AD due to the lack of effective treatment for the disease. AD progresses slowly before clinical biomarkers appear.

""",unsafe_allow_html=True)
    
    c2.image("./img/Alzheimer6.jpg")
    c2.markdown("""
                
In our quest to aid early diagnosis and intervention, we have developed a prediction model that utilizes advanced technology to analyze MRI (Magnetic Resonance Imaging) scans of the brain. MRI scans provide detailed images of the brain's structure and function, allowing us to identify subtle changes associated with Alzheimer's disease.                

Our prediction model harnesses the power of these MRI scans to assess the risk of developing Alzheimer's disease in individuals. By analyzing specific features and patterns in the brain captured by MRI, the model can provide valuable insights into a person's likelihood of experiencing cognitive decline associated with Alzheimer's disease.

In the following sections, we'll delve into how our prediction model works, the significance of early detection, and how patients can understand the results of their MRI scan predictions. We aim to empower individuals and their families with knowledge and resources to navigate their journey with Alzheimer's disease more effectively.

--- 


""",unsafe_allow_html=True)
    
    c1.image("./img/Alzheimer4.jpg")


    st.markdown(horizontal_bar, True) 

    st.markdown("# **prediction of MRI brain**", unsafe_allow_html=True)
    with st.expander("let's go "):
        
        st.markdown("""
    ### Classification of MRI brain image to know type of Alzheimer's disease
    ---
                                
    """ , unsafe_allow_html=True)
        st.title("ResNet-50 Image Classifier :tea: :coffee:")
        upload = st.file_uploader(label="Upload Image :", type=["png", "jpg", "jpeg"])

        if upload:
            img = Image.open(upload)

            processed_img = preprocess_func(img)
            model = load_model()
            probs, idxs = make_prediction(model, processed_img)
            feature_imp = interpret_prediction(model, processed_img, idxs[0])

            interp_fig, ax = viz.visualize_image_attr(feature_imp, show_colorbar=True, fig_size=(6,6))

            prob_fig = plt.figure(figsize=(12,2.5))
            ax = prob_fig.add_subplot(111)
            plt.barh(y=categories[idxs][::-1], width=probs[::-1], color=["dodgerblue"]*4+["tomato"])
            plt.title('Top 5 Probabilities', loc="center", fontsize=15)
            st.pyplot(prob_fig, use_container_width=True)

            col1, col2 = st.columns(2, gap="medium")

            with col1:
                main_fig = plt.figure(figsize=(6,6))
                ax = main_fig.add_subplot(111)
                plt.imshow(img);
                plt.xticks([],[]);
                plt.yticks([],[]);
                st.pyplot(main_fig, use_container_width=True)

            with col2:
                st.pyplot(interp_fig, use_container_width=True)
            
            

    st.markdown(horizontal_bar, True) 

    st.markdown("""
### Explanation of MRI Scans

##### *What is an MRI Scan??*

MRI stands for Magnetic Resonance Imaging. It's a safe and non-invasive technique that doctors use to get detailed images of the inside of your body, including your brain.
Think of an MRI scan like taking a photograph of your brain, but instead of using a camera, it uses powerful magnets and radio waves.

##### *How Does It Work?*

The MRI machine creates a strong magnetic field around your head. This field, along with radio waves, interacts with the water molecules in your brain to produce signals.
These signals are captured by the machine and processed by a computer to create detailed images of your brain's structure and function.

##### *Why is it Important?*

These images help doctors see changes in your brain that might be related to Alzheimer's disease. By looking at these images, doctors can identify patterns that suggest whether Alzheimer's might be present.

                
##### *Can an MRI Diagnose Alzheimer’s?*

The simplest answer to the question is yes. The more complicated answer considers that there is still a lot of research to do on this disease, so it may be a while before we establish a definitive test to diagnose Alzheimer’s disease.
However, for the time being, using an MRI to detect Alzheimer’s is one of the best options available.

##### Easy to Understand Analogy:

Imagine your brain is like a city. An MRI scan is like taking a high-resolution aerial photo of that city, allowing you to see the roads, buildings, and parks clearly. Similarly, an MRI gives doctors a clear picture of the different parts of your brain and how they're working.


""" , unsafe_allow_html=True)
    st.image("./img/MRI.jpg")
    st.markdown(horizontal_bar, True) 

    

    st.markdown("""
### Role of MRI Scans in Predicting Alzheimer's Disease

**How MRI Scans Help:**
- MRI scans give doctors a detailed look at your brain's structure and health. These images show different parts of your brain and how they're working.
- When it comes to Alzheimer's disease, certain changes in the brain can be early signs of the condition. MRI scans can help detect these changes.
""" , unsafe_allow_html=True)
    
    c3,c4 = st.columns(2)
    c3.markdown("""
**What the Prediction Model Looks For:**
- **Brain Volume:** One of the things the prediction model looks at is the size of different parts of your brain. In Alzheimer's disease, some areas of the brain may shrink. The model can detect these changes in brain volume.
- **Abnormal Protein Deposits:** The model also looks for signs of abnormal protein deposits, like amyloid plaques and tau tangles, which are often found in the brains of people with Alzheimer's.
- **Brain Connectivity:** The model examines how different parts of your brain communicate with each other. Changes in brain connectivity can be another sign of Alzheimer's disease.
""" , unsafe_allow_html=True)
    
    c4.image("./img/Alzheimer5.jpg")
    
    st.markdown("""
**How It Works:**
- The prediction model uses advanced algorithms to analyze the MRI scans. Think of it like a very smart detective that looks for clues in the images.
- By identifying specific patterns and features in the MRI scans, the model can estimate the likelihood that a person might have or develop Alzheimer's disease.

**Why This Matters:**
- Understanding these changes early on allows doctors to make informed decisions about your health. Early detection can lead to better planning and treatment options, potentially slowing the progression of the disease and improving your quality of life.


""" , unsafe_allow_html=True)
    
    st.markdown(horizontal_bar, True) 

    st.markdown("""
### Key Factors Considered by the Prediction Model

**Understanding the Important Clues:**
- The prediction model looks at specific features in your MRI scans that are linked to Alzheimer's disease. These features, also known as biomarkers, help the model assess your risk.

**Key Biomarkers:**
""" , unsafe_allow_html=True)

    c5,c6 = st.columns(2)

    c5.markdown("""
1. **Hippocampal Volume:**
   - The hippocampus is a part of the brain that's crucial for memory. In Alzheimer's disease, the hippocampus often shrinks. The model checks the size of the hippocampus to see if there's any reduction in volume.

2. **Cortical Thickness:**
   - The cortex is the brain's outer layer, responsible for many important functions like thinking, remembering, and speaking. Thinning of the cortex can be a sign of Alzheimer's. The model measures the thickness of the cortex in different areas of the brain.

3. **White Matter Integrity:**
   - White matter consists of nerve fibers that connect different parts of the brain. Damage or changes in white matter can affect brain function. The model assesses the integrity of white matter to detect any abnormalities.

4. **Presence of Amyloid Plaques:**
   - Amyloid plaques are abnormal protein deposits found in the brains of people with Alzheimer's disease. These plaques can disrupt brain function. The model looks for signs of amyloid plaques in the MRI scans.
""" , unsafe_allow_html=True)
    
    c6.markdown("""
5. **Neurofibrillary Tangles:**
   - Neurofibrillary tangles are twisted fibers found inside brain cells. They are another hallmark of Alzheimer's disease. The model checks for the presence of these tangles.
""" , unsafe_allow_html=True)
    
    c6.image("./img/Alzheimer3.jpg")


    st.markdown("""
**Why These Factors Matter:**
- These biomarkers are known to be associated with the development and progression of Alzheimer's disease. By analyzing these factors, the model can make informed predictions about your risk.
- Understanding these key factors helps doctors provide you with a more accurate assessment and tailor your care accordingly.


""" , unsafe_allow_html=True)
    
    st.markdown(horizontal_bar, True) 

    st.markdown("""
### Understanding the Prediction Process

**How the Prediction Model Works:**
- The prediction model uses advanced technology to analyze your MRI scans and assess your risk for Alzheimer's disease. Let's break down the process into simple steps:

**1. Uploading the MRI Scan:**
   - The process starts with taking an MRI scan of your brain. This scan is a detailed image that shows the structure and condition of your brain.

**2. Analyzing the Scan:**
   - The MRI scan is then fed into the prediction model. Think of the model as a highly skilled detective that looks for specific clues in the scan.
   - The model examines the MRI scan to identify key biomarkers, which are features in the brain that can indicate the presence of Alzheimer's disease.

**3. Identifying Biomarkers:**
   - The model looks for changes or abnormalities in important areas of the brain. For example, it checks:
     - **Hippocampal Volume:** To see if the hippocampus, a critical memory area, has shrunk.
     - **Cortical Thickness:** To measure if the outer layer of the brain is thinning.
     - **White Matter Integrity:** To ensure the nerve fibers connecting different parts of the brain are healthy.
     - **Amyloid Plaques:** To detect any abnormal protein deposits.
     - **Neurofibrillary Tangles:** To look for twisted fibers inside brain cells.
""" , unsafe_allow_html=True)
    
    c7,c8 = st.columns(2)

    c7.markdown("""           

**4. Generating Predictions:**
   - After analyzing the scan, the model combines all the information about the biomarkers. It uses this data to calculate the likelihood of developing Alzheimer's disease.
   - The model then generates a prediction, which can indicate a high, moderate, or low risk of Alzheimer's disease.

**5. Sharing Results:**
   - The prediction results are shared with your doctor. These results help your doctor understand your brain health better and decide on the next steps in your care plan.
""" , unsafe_allow_html=True)

    
     
    c8.image("./img/Alzheimer7.jpg")
    st.markdown("""    
**Simple and Clear Explanations:**
- We ensure the entire process is explained in a way that's easy to understand. Our goal is to help you grasp how the model works without overwhelming you with technical terms.
- If you have any questions or need further clarification, our team is here to help you at every step.    
                
""" , unsafe_allow_html=True)

    st.markdown(horizontal_bar, True)

    st.markdown("""    
### Closing Thoughts

**Summarizing the Key Points:**
- Understanding how MRI scans are used to predict Alzheimer's disease can empower you to take proactive steps in managing your health. Here’s a quick recap of the key points:
  - **MRI Scans:** They are a safe, non-invasive way to get detailed images of your brain.
  - **Prediction Model:** Uses these MRI scans to identify important biomarkers associated with Alzheimer's disease.
  - **Early Detection:** Helps in taking early action to manage the disease and improve your quality of life.

**The Importance of MRI Scans in Predicting Alzheimer’s Disease:**
- MRI scans provide crucial information that can help in the early detection of Alzheimer's disease.
- Early detection through MRI scans allows for timely interventions, which can slow the progression of the disease and enhance treatment effectiveness.

**Encouraging Open Communication:**
- It’s important to stay informed and proactive about your health.
- If you have any questions or concerns about the prediction process or the results, don’t hesitate to discuss them with your healthcare provider.
- Your doctor can provide additional insights, answer your questions, and help you understand the next steps in your care journey.
""" , unsafe_allow_html=True)

    c9,c10 = st.columns(2)

    c9.markdown("""  
**Empowering Your Health Journey:**
- Knowledge is power. By understanding how MRI scans and prediction models work, you are taking an important step towards better managing your health.
- Always feel free to reach out to support resources, join support groups, or access additional educational materials to stay informed and supported.
               
""" , unsafe_allow_html=True)
    
    c10.image("./img/Alzheimer3.png")

    st.markdown(horizontal_bar, True)
