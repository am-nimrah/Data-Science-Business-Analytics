import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

# Load dataset
def load_data():
    df = pd.read_csv("StudentsPerformance.csv")
    return df

# Preprocess data
def preprocess_data(df):
    label_encoders = {}
    categorical_columns = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
    
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # Store encoders

    return df, label_encoders

# Train and save model
def train_model():
    df = load_data()
    df, label_encoders = preprocess_data(df)

    # Select features and target
    X = df[['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course', 'reading score', 'writing score']]
    y = df['math score']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save model and encoders
    joblib.dump(model, "student_model.pkl")
    joblib.dump(label_encoders, "label_encoders.pkl")

# Load trained model and encoders
def load_model():
    model = joblib.load("student_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    return model, label_encoders

if __name__ == "__main__":
    train_model()
