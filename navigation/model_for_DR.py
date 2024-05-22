import streamlit as st

def model_page():
    


    st.markdown("""
### Welcome, Doctors!

#### Understanding Our Prediction Model

Welcome to our webpage, where we offer a comprehensive overview of our cutting-edge prediction model for diagnosing Alzheimer's disease. As medical professionals, we understand the importance of clarity and accuracy when it comes to utilizing predictive tools for patient care.

#### What You'll Find Here:

1. **Model Overview**: We provide detailed insights into how our prediction model works, including its underlying architecture, data sources, and the methodologies employed in its development.

2. **Accuracy and Validation**: Learn about the rigorous validation process undertaken to ensure the accuracy and reliability of our model's predictions. We present evidence of its performance and discuss its clinical implications.

3. **Clinical Relevance**: Explore how our prediction model can aid in early diagnosis and treatment planning for Alzheimer's disease. We highlight its potential impact on clinical practice and patient care.

#### Why It Matters:

Our prediction model isn't just about numbers and algorithms—it's about improving patient outcomes and empowering healthcare professionals like you to make informed decisions. By understanding how our model works and its clinical implications, you can better integrate it into your practice and provide optimal care for your patients.

#### Join Us in the Fight Against Alzheimer's:

Alzheimer's disease is a complex condition with significant implications for patients and their families. With your support and collaboration, we can continue to refine and optimize our prediction model, ultimately making a meaningful difference in the lives of those affected by this devastating disease.

#### Explore Further:

Ready to dive deeper into our prediction model? Click below to access detailed information, validation studies, and resources to support its integration into your clinical practice.

---

Feel free to adjust the tone and content to better align with your webpage's style and audience preferences.
""",unsafe_allow_html=True)
    st.subheader("🔎🪐Overview of the Prediction Model:")
    with st.expander ("1. Model Architecture") :
        st.markdown ("""
## 1. Model Architecture:                     
#### Swin Transformer:
The prediction model leverages the Swin Transformer architecture, a cutting-edge deep learning model that has shown remarkable performance in various computer vision tasks.

#### Key Components:

##### a. Patch Embedding:
The Swin Transformer breaks down the input image into smaller patches, treating each patch as a token. These patches are then linearly projected into a lower-dimensional space to create embeddings. This process allows the model to efficiently process large images by focusing on local regions.

##### b. Multi-Scale Self-Attention:
One of the distinguishing features of the Swin Transformer is its multi-scale self-attention mechanism. Instead of computing self-attention across all tokens simultaneously, the model organizes tokens into hierarchical groups called "windows" and performs self-attention within each window. This approach enables the model to capture both local and global dependencies in the input image efficiently.

##### c. Hierarchical Representation Learning:
By organizing tokens into a hierarchical structure, the Swin Transformer facilitates hierarchical representation learning. Tokens at different hierarchical levels capture features at varying spatial resolutions, allowing the model to extract rich and informative representations from the input image. This hierarchical approach enhances the model's ability to understand complex patterns and relationships within the data.

##### d. Feedforward Networks:
In addition to self-attention layers, the Swin Transformer includes feedforward networks within each transformer block. These networks process token embeddings independently and apply non-linear transformations, enabling the model to capture complex relationships between tokens.

##### e. Layer Norm and Residual Connections:
To stabilize training and facilitate gradient flow, the Swin Transformer incorporates layer normalization and residual connections within each transformer block. These mechanisms help mitigate the vanishing gradient problem and improve the model's ability to learn meaningful representations from the data.

###### By utilizing the Swin Transformer architecture, the prediction model achieves state-of-the-art performance in predicting Alzheimer's disease based on MRI scans. The combination of patch embedding, multi-scale self-attention, and hierarchical representation learning enables the model to effectively capture and analyze complex patterns in medical imaging data, leading to accurate and reliable predictions.
          """ , unsafe_allow_html=True)
        st.image("./project4/img/swin.png")

    with st.expander("2. Training Data") :
        st.markdown("""
## 2. Training Data:

#### Alzheimer's Dataset:
The prediction model is trained on the Alzheimer's Dataset, a curated collection of MRI scans obtained from patients diagnosed with various stages of Alzheimer's disease. This dataset serves as the foundation for training the model to recognize patterns and features indicative of Alzheimer's disease progression.

#### Classes in the Dataset:
The Alzheimer's Dataset comprises MRI scans from individuals diagnosed with different stages of Alzheimer's disease. These stages are categorized into distinct classes, each representing a specific level of disease severity. The classes present in the dataset include:

1. **MildDemented**: This class represents MRI scans from patients diagnosed with mild dementia due to Alzheimer's disease. These individuals typically exhibit early symptoms of cognitive decline, including memory loss and difficulty with daily tasks.

2. **NonDemented**: MRI scans belonging to this class are obtained from individuals who do not exhibit signs of dementia or cognitive impairment. These scans serve as a reference for normal brain structure and function, providing a baseline for comparison.

3. **VeryMildDemented**: Individuals in this class have MRI scans showing very mild or early-stage symptoms of dementia related to Alzheimer's disease. While cognitive impairment may be present, it is often subtle and may not significantly impact daily functioning.

###### By training the model on this diverse dataset with multiple classes representing different stages of Alzheimer's disease, the prediction model learns to distinguish between normal brain scans and those indicative of Alzheimer's disease progression. This training process enables the model to make accurate predictions when presented with new MRI scans, aiding in the early detection and diagnosis of Alzheimer's disease.
""" , unsafe_allow_html=True)   
        st.image("./project4/img/MRI.png")

    with st.expander("3. Data Preprocessing") :
        st.markdown("""
## 3. Data Preprocessing:

#### Preprocessing Steps:
Before feeding the MRI scans into the prediction model, several preprocessing steps are applied to ensure optimal model performance and generalization. These steps typically include:

1. **Resizing**:
   - The MRI scans are resized to a standardized resolution to ensure uniformity in input dimensions. Resizing helps facilitate model training by ensuring that all images have the same size, reducing computational complexity and memory requirements.

2. **Center Cropping**:
   - Center cropping involves extracting a central region of interest from the resized MRI scans. This process helps focus the model's attention on the most relevant anatomical structures within the image, improving its ability to extract meaningful features.

3. **Normalization**:
   - Normalization is performed to standardize the intensity values of the MRI scans across different images. This typically involves scaling the pixel values to a common range (e.g., [0, 1] or [-1, 1]) to make the data more amenable to training. Normalization helps mitigate variations in image intensity caused by differences in acquisition protocols or scanner settings.

#### Data Augmentation:
In addition to preprocessing, data augmentation techniques are employed to enhance model generalization and robustness. These techniques introduce variations to the MRI scans, creating synthetic training examples without altering the underlying pathology. Common data augmentation techniques include:

1. **Random Horizontal/Vertical Flips**:
   - MRI scans may be horizontally or vertically flipped with a certain probability during training. This augmentation technique introduces variations in orientation, helping the model learn invariant features that are not dependent on the specific orientation of the input image.

2. **Color Jittering**:
   - Color jittering involves randomly perturbing the color channels of the MRI scans by applying transformations such as brightness, contrast, saturation, and hue adjustments. This augmentation technique helps the model become more robust to variations in image appearance, such as differences in lighting conditions or contrast levels.

###### By incorporating these preprocessing steps and data augmentation techniques, the prediction model is exposed to a more diverse and representative training dataset. This helps improve its ability to generalize to unseen MRI scans and enhances its performance in accurately predicting Alzheimer's disease based on the input images.
""" , unsafe_allow_html=True)

    with st.expander("4. Model Fine-tuning") :
        st.markdown("""
## 4. Model Fine-tuning:

#### Process Outline:
Fine-tuning a pre-trained Swin Transformer model for the specific task of Alzheimer's disease prediction involves adapting the weights of the pre-trained model to better fit the characteristics of the target dataset. The process typically follows these steps:

1. **Loading Pre-trained Model**:
   - Initially, a pre-trained Swin Transformer model is loaded. This model has been previously trained on a large-scale dataset (e.g., ImageNet) to learn generic visual representations.

2. **Freezing Pre-trained Layers**:
   - To prevent the pre-trained weights from being overly adjusted during fine-tuning and to retain the learned features, the parameters of the early layers in the Swin Transformer model are often frozen. This means that these layers are not updated during the fine-tuning process.

3. **Modifying Model Head**:
   - The final layers of the Swin Transformer model, often referred to as the "head," are replaced or modified to adapt the model to the specific task of Alzheimer's disease prediction. This modification typically involves adding new fully connected layers followed by activation functions and dropout regularization.

4. **Training on Task-specific Data**:
   - The modified Swin Transformer model is then trained on the task-specific dataset, which in this case consists of MRI scans from individuals diagnosed with different stages of Alzheimer's disease. During training, the model learns to extract relevant features from the MRI scans and predict the corresponding disease labels.

5. **Fine-tuning Parameters**:
   - Throughout the training process, the parameters of the modified layers (i.e., the model head) are fine-tuned using backpropagation and gradient descent optimization. The objective is to minimize a loss function that quantifies the discrepancy between the predicted labels and the ground truth labels in the training dataset.

6. **Regularization and Optimization**:
   - Techniques such as dropout regularization may be applied during training to prevent overfitting and improve model generalization. Additionally, optimization algorithms such as AdamW may be used to update the model parameters efficiently.

#### Head Modification:
In the context of fine-tuning for Alzheimer's disease prediction, the head of the Swin Transformer model is typically modified by:

1. **Adding Fully Connected Layers**:
   - New fully connected layers are appended to the existing head of the Swin Transformer model. These layers serve as a means to learn task-specific representations from the features extracted by the pre-trained layers.

2. **ReLU Activation**:
   - Rectified Linear Unit (ReLU) activation functions are commonly applied after the fully connected layers to introduce non-linearity into the model and enable it to learn complex relationships between features.

3. **Dropout Regularization**:
   - Dropout regularization is often employed between the fully connected layers to prevent overfitting. Dropout randomly drops a certain proportion of the neurons during training, forcing the model to learn more robust and generalizable representations.

###### By fine-tuning the pre-trained Swin Transformer model and modifying its head architecture, the model can adapt to the task of Alzheimer's disease prediction and achieve improved performance on the target dataset.
""" , unsafe_allow_html=True)
        st.image("./project4/img/model.png")

    with st.expander("5. Loss Function and Optimization") :
        st.markdown("""
## 5. Loss Function and Optimization:

#### Loss Function:
The loss function used for training the model is crucial as it quantifies the discrepancy between the predicted output of the model and the actual target labels. In this case, the loss function chosen is LabelSmoothingCrossEntropy.
    
                    """,unsafe_allow_html=True)
        st.image("./project4/img/loss.png") 
        st.markdown("""                    
##### LabelSmoothingCrossEntropy:
LabelSmoothingCrossEntropy is a variant of the traditional cross-entropy loss function, commonly used in classification tasks. It addresses the issue of overconfidence by penalizing the model for being too confident in its predictions. Specifically, it smooths the ground truth labels by redistributing some of the probability mass from the true class to other classes. This regularization encourages the model to learn more robust decision boundaries and reduces the risk of overfitting.

#### Optimization Algorithm:
The optimization algorithm is responsible for updating the parameters of the model during training to minimize the chosen loss function. In this case, the optimization algorithm employed is AdamW.

##### AdamW:
AdamW is a variant of the Adam optimization algorithm, which stands for Adaptive Moment Estimation with Weight Decay. It combines the benefits of adaptive learning rates and momentum-based updates to efficiently optimize the model parameters. The addition of weight decay regularization (hence the "W" in AdamW) further helps prevent overfitting by penalizing large parameter values. AdamW is known for its effectiveness in training deep neural networks and is widely used in various machine learning tasks.

#### Learning Rate Scheduling:
Learning rate scheduling strategies may also be employed to adjust the learning rate during training dynamically. This helps improve convergence and prevents the optimization process from getting stuck in local minima. While the specific scheduling strategy used is not mentioned, common approaches include:

- **StepLR:** Reducing the learning rate by a factor (gamma) after a certain number of epochs.
- **ExponentialLR:** Decaying the learning rate exponentially over time.
- **ReduceLROnPlateau:** Adjusting the learning rate based on the validation loss plateau.

###### These scheduling strategies ensure that the model's learning rate adapts to the training dynamics, leading to more stable and efficient optimization.

###### By using LabelSmoothingCrossEntropy as the loss function, AdamW as the optimization algorithm, and potentially employing learning rate scheduling strategies, the model training process is optimized to efficiently learn from the data and generalize well to unseen examples.
""" , unsafe_allow_html=True)

    with st.expander("6. Training and Evaluation") :
        st.markdown("""
## 6. Training and Evaluation:

#### Training Process:
The training process involves iteratively updating the parameters of the model using the training dataset to minimize the chosen loss function. Key insights into the training process include:

1. **Number of Epochs Trained**:
   - This refers to the total number of passes through the entire training dataset during training. Each epoch consists of one forward pass (computing predictions), one backward pass (computing gradients), and one update step (updating model parameters). The number of epochs trained depends on factors such as dataset size, model complexity, and convergence criteria.

2. **Training/Validation Splits**:
   - The dataset is typically split into training and validation sets to monitor the model's performance during training and prevent overfitting. The training set is used to update the model parameters, while the validation set is used to evaluate the model's performance on unseen data and tune hyperparameters.

#### Model Evaluation:
Evaluation metrics are used to assess the performance of the trained model on both the validation and test sets.

1. **Validation Set Accuracy**:
   - This metric quantifies the accuracy of the model's predictions on the validation set, which consists of data that the model has not seen during training. It provides insights into how well the model generalizes to unseen data and helps identify potential overfitting.

2. **Improvements Over Epochs**:
   - Monitoring the validation set accuracy over epochs allows us to track the model's learning progress and identify improvements or stagnation. Observing increases in validation accuracy over epochs indicates that the model is learning relevant patterns from the data and improving its performance.

3. **Test Set Evaluation**:
   - Once training is complete, the trained model is evaluated on an independent test set to assess its performance in real-world scenarios. The test set provides an unbiased estimate of the model's generalization ability and its performance on unseen data. Evaluation metrics such as accuracy, precision, recall, F1-score, and confusion matrix may be computed to analyze the model's performance comprehensively.

#### Overall Performance:
The evaluation results on the test set provide insights into the overall performance of the trained model. By comparing the predicted labels with the ground truth labels in the test set, we can assess the model's accuracy, robustness, and effectiveness in predicting Alzheimer's disease based on MRI scans.

###### In summary, the training and evaluation process involves training the model on the training set, monitoring its performance on the validation set, and finally evaluating its performance on the independent test set to ensure reliable and accurate predictions in real-world scenarios.
""" , unsafe_allow_html=True)

    with st.expander("7. Ensemble Model") :
        st.markdown("""
## 7. Ensemble Model:

#### Creation of Ensemble Model:
In the context of improving prediction accuracy, an ensemble model is created by combining predictions from multiple individual models. In this case, an ensemble model is formed by combining the predictions of two distinct models: the Swin Transformer model and a pre-trained ResNet50 model.

1. **Swin Transformer Model**:
   - The Swin Transformer model is a deep learning architecture specifically designed for computer vision tasks, including image classification. It has its strengths and weaknesses in capturing different aspects of the input data.

2. **Pre-trained ResNet50 Model**:
   - ResNet50 is another popular deep learning architecture commonly used for image classification tasks. It utilizes residual connections to address the vanishing gradient problem and has demonstrated strong performance on various datasets.

#### Combining Predictions:
The ensemble model combines predictions from both the Swin Transformer model and the pre-trained ResNet50 model to make final predictions. This combination is typically achieved through a simple averaging or voting mechanism:

1. **Averaging Predictions**:
   - Each individual model independently predicts the class probabilities or labels for a given input image. These predictions are then averaged across both models to obtain a final set of class probabilities. The class with the highest average probability is selected as the final prediction.

2. **Voting Mechanism**:
   - Alternatively, each individual model casts a "vote" for its predicted class label. The final prediction is determined by a majority voting scheme, where the class with the most votes across both models is selected as the final prediction.

#### Benefits of Ensemble Modeling:
Ensemble modeling offers several advantages over using a single model alone:

- **Improved Prediction Accuracy**: By leveraging the strengths of multiple models, ensemble modeling can often achieve higher prediction accuracy compared to individual models.
- **Robustness to Model Variability**: Ensemble models are less susceptible to overfitting and model variability, as they combine predictions from multiple diverse models.
- **Enhanced Generalization**: Ensemble models tend to generalize well to unseen data, as they capture a broader range of patterns and features present in the dataset.

###### Overall, the ensemble model combining the Swin Transformer model with a pre-trained ResNet50 model offers an effective approach to improving prediction accuracy and robustness in Alzheimer's disease prediction based on MRI scans.
""" , unsafe_allow_html=True)
        
    with st.expander("8. Conclusion and Future Directions") :
        st.markdown("""
## 8. Conclusion and Future Directions:

#### Significance of the Prediction Model:
The conclusion highlights the critical role of the developed prediction model in facilitating early diagnosis and informing treatment planning for Alzheimer's disease. Key points to emphasize include:

1. **Early Diagnosis**: The prediction model serves as a valuable tool for early detection of Alzheimer's disease based on MRI scans, enabling timely intervention and improved patient outcomes.
  
2. **Treatment Planning**: By accurately predicting the presence or progression of Alzheimer's disease, the model aids clinicians in developing personalized treatment plans tailored to the individual needs of patients.

3. **Clinical Impact**: Emphasize the potential clinical impact of the model in enhancing healthcare delivery, optimizing resource allocation, and ultimately improving the quality of life for individuals affected by Alzheimer's disease.

#### Future Directions:
In addition to summarizing the achievements of the current prediction model, the conclusion also outlines potential future directions for research and development. This section explores avenues for further improvement and innovation, including:

1. **Incorporating Additional Data Sources**:
   - Explore the integration of diverse data sources, such as genetic information, cognitive assessments, and biomarkers, to enhance the predictive capabilities of the model and provide a more comprehensive understanding of Alzheimer's disease.

2. **Advanced Model Architectures**:
   - Investigate the adoption of advanced model architectures, such as graph neural networks or attention mechanisms, to capture intricate relationships within MRI data and improve prediction accuracy.

3. **Longitudinal Analysis**:
   - Conduct longitudinal analysis by tracking changes in MRI scans over time to predict disease progression and evaluate the effectiveness of treatment interventions.

4. **Interpretability and Explainability**:
   - Enhance the interpretability and explainability of the model predictions to facilitate clinical decision-making and foster trust among healthcare practitioners.

5. **Clinical Validation**:
   - Conduct rigorous clinical validation studies to assess the real-world performance of the prediction model across diverse patient populations and clinical settings.

6. **Translation to Clinical Practice**:
   - Explore strategies for integrating the prediction model into routine clinical practice, including user-friendly interfaces, decision support systems, and validation within healthcare workflows.

###### By outlining these potential future directions, the conclusion sets the stage for ongoing research and innovation in the field of Alzheimer's disease prediction, with the ultimate goal of improving patient care and advancing our understanding of this complex neurological condition.
""" , unsafe_allow_html=True)
        st.image("./project4/img/nnModel.png")  #https://www.nature.com/articles/s41598-020-79243-9


        














