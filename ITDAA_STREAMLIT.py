#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import streamlit as st
import pandas as pd
import joblib

# Load the saved model
model = joblib.load('best_model.pkl')

# Define the user interface
def user_interface():
    st.title('Heart Disease Prediction')

    st.write('Please enter the following details for the patient:')

    age = st.number_input('Age', min_value=0, max_value=120, value=25)
    sex = st.selectbox('Sex', ['Male', 'Female'])
    cp = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'])
    trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=0, max_value=300, value=120)
    chol = st.number_input('Serum Cholesterol (mg/dl)', min_value=0, max_value=600, value=200)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['True', 'False'])
    restecg = st.selectbox('Resting Electrocardiographic Results', ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'])
    thalach = st.number_input('Maximum Heart Rate Achieved', min_value=0, max_value=300, value=150)
    exang = st.selectbox('Exercise Induced Angina', ['Yes', 'No'])
    oldpeak = st.number_input('ST Depression Induced by Exercise Relative to Rest', min_value=0.0, max_value=10.0, value=0.0)
    slope = st.selectbox('Slope of the Peak Exercise ST Segment', ['Upsloping', 'Flat', 'Downsloping'])
    ca = st.number_input('Number of Major Vessels Colored by Flouroscopy', min_value=0, max_value=4, value=0)
    thal = st.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'])

    if st.button('Predict'):
        # Prepare data for prediction
        patient_data = {
            'age': age,
            'sex': 1 if sex == 'Male' else 0,
            'cp': ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'].index(cp),
            'trestbps': trestbps,
            'chol': chol,
            'fbs': 1 if fbs == 'True' else 0,
            'restecg': ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'].index(restecg),
            'thalach': thalach,
            'exang': 1 if exang == 'Yes' else 0,
            'oldpeak': oldpeak,
            'slope': ['Upsloping', 'Flat', 'Downsloping'].index(slope),
            'ca': ca,
            'thal': ['Normal', 'Fixed Defect', 'Reversible Defect'].index(thal)
        }

        # Make prediction
        prediction = model.predict(pd.DataFrame([patient_data]))
        if prediction[0] == 1:
            st.error('The patient is likely to have heart disease.')
        else:
            st.success('The patient is unlikely to have heart disease.')

# Run the application
if __name__ == '__main__':
    user_interface()


# In[ ]:




