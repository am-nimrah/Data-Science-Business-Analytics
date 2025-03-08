import streamlit as st
import numpy as np
import joblib
from model import load_model

# Load trained model and encoders
model, label_encoders = load_model()

# Custom CSS styling
st.markdown("""
    <style>
    .big-font { font-size:30px !important; font-weight: bold; }
    .stButton>button { background-color: #4CAF50; color: white; padding: 10px 24px; border-radius: 8px; border: none; font-size: 16px; }
    .stButton>button:hover { background-color: #45a049; }
    .stSuccess { font-size: 20px; color: #4CAF50; }
    .footer { font-size: 16px; text-align: center; margin-top: 20px; }
    .project-link { font-size: 18px; color: #1e90ff; text-decoration: none; }
    .project-link:hover { text-decoration: underline; }
    </style>
    """, unsafe_allow_html=True)

# UI layout
st.title("📊 Student Performance Prediction")
st.markdown('<p class="big-font">Predict the <strong>Math Score</strong> based on student attributes.</p>', unsafe_allow_html=True)

# User inputs
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", label_encoders['gender'].classes_)
    race = st.selectbox("Race/Ethnicity", label_encoders['race/ethnicity'].classes_)
    parent_education = st.selectbox("Parental Level of Education", label_encoders['parental level of education'].classes_)

with col2:
    lunch = st.selectbox("Lunch Type", label_encoders['lunch'].classes_)
    prep_course = st.selectbox("Test Preparation Course", label_encoders['test preparation course'].classes_)
    reading_score = st.slider("Reading Score", 0, 100, 50)
    writing_score = st.slider("Writing Score", 0, 100, 50)

# Encode user inputs
input_data = np.array([
    label_encoders['gender'].transform([gender])[0],
    label_encoders['race/ethnicity'].transform([race])[0],
    label_encoders['parental level of education'].transform([parent_education])[0],
    label_encoders['lunch'].transform([lunch])[0],
    label_encoders['test preparation course'].transform([prep_course])[0],
    reading_score,
    writing_score
]).reshape(1, -1)

# Prediction
if st.button("Predict Score"):
    predicted_score = model.predict(input_data)[0]
    st.markdown(f'<p class="stSuccess">📈 Predicted Math Score: <strong>{predicted_score:.2f}</strong></p>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown('<p class="footer">Developed by <strong>Nimra Waqar</strong> 🚀</p>', unsafe_allow_html=True)
