import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("StudentsPerformance.csv")
    return df

df = load_data()

# Preprocessing: Encode categorical variables
def preprocess_data(df):
    label_encoders = {}
    categorical_columns = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
    
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # Store encoders for later use

    return df, label_encoders

df, label_encoders = preprocess_data(df)

# Feature selection
X = df[['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course', 'reading score', 'writing score']]
y = df['math score']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "student_model.pkl")

# Load trained model
model = joblib.load("student_model.pkl")

# Custom CSS for styling
st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stSelectbox, .stSlider {
        margin-bottom: 20px;
    }
    .stSuccess {
        font-size: 20px;
        color: #4CAF50;
    }
    .stMarkdown {
        margin-top: 20px;
    }
    .footer {
        font-size: 16px;
        text-align: center;
        margin-top: 20px;
    }
    .project-link {
        font-size: 18px;
        color: #1e90ff;
        text-decoration: none;
    }
    .project-link:hover {
        text-decoration: underline;
    }
    </style>
    """, unsafe_allow_html=True)

# Author and Affiliation Section
st.markdown("""
    <div style="text-align: center;">
        <p>👩‍💻 <strong>Author:</strong> <a href="https://www.linkedin.com/in/nimrah-waqar" target="_blank">Nimra Waqar</a></p>
        <p>🏢 <strong>Associated with:</strong> <a href="https://thesparksfoundationsingapore.org/" target="_blank">The Sparks Foundation</a></p>
    </div>
    """, unsafe_allow_html=True)

# Streamlit UI
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

# Show sample dataset
if st.checkbox("Show Dataset"):
    st.dataframe(df.head())

# Other Projects Section
st.markdown("---")
st.markdown("### Other Projects")
st.markdown("""
    <p>Check out my other project on <strong>Iris Dataset PCA Analysis</strong>:</p>
    <a class="project-link" href="https://iris-pca-spf-nimra.streamlit.app/" target="_blank">🌼 Iris PCA Analysis</a>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown('<p class="footer">Developed by <strong>Nimra Waqar</strong> 🚀</p>', unsafe_allow_html=True)