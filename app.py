import streamlit as st
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Page config
st.set_page_config(page_title=" Salary Prediction", layout="centered")

# Glassmorphism + animated gradient background CSS
def set_glassmorphism_style():
    css = """
    <style>
    /* Animate background for body and Streamlit app wrapper */
    body, .stApp {
        height: 100vh !important;
        margin: 0 !important;
        background: linear-gradient(-45deg, #c1f0dc, #e3d7ff, #d9fdf2, #f3e8ff);
        background-size: 400% 400%;
        animation: gradientBG 20s ease infinite;
        font-family: 'Segoe UI', sans-serif;
    }

    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Glassmorphism effect on main content container */
    .main .block-container {
        background: rgba(255, 255, 255, 0.3) !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }

    /* Text color for headings and labels */
    h1, h2, h3, h4, h5, h6, p, label {
        color: #333 !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_glassmorphism_style()

# Title and intro
st.title("💸 Salary Prediction App")
st.write("Enter employee details below to predict salary:")

# Load trained model
model = pickle.load(open("linear_model.pkl", "rb"))

# Load dataset for performance metrics
df = pd.read_csv("Salary Data.csv").dropna()

# Encode categorical columns using LabelEncoder
le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])
le_education = LabelEncoder()
df['Education Level'] = le_education.fit_transform(df['Education Level'])
le_job = LabelEncoder()
df['Job Title'] = le_job.fit_transform(df['Job Title'])

X = df.drop("Salary", axis=1)
y = df["Salary"]

# Train/test split for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Dropdown options matching your data (adjust if needed)
education_options = ['High School', 'Bachelor', 'Master', 'PhD']
job_title_options = ['Data Scientist', 'Software Engineer', 'Manager', 'Analyst']
gender_options = ['Male', 'Female', 'Other']

# Input widgets
age = st.slider("Age", 18, 70, 30)
gender = st.selectbox("Gender", gender_options)
education = st.selectbox("Education Level", education_options)
job_title = st.selectbox("Job Title", job_title_options)
experience = st.slider("Years of Experience", 0, 50, 5)

# Maps for encoding user input
gender_map = {'Male': 0, 'Female': 1, 'Other': 2}
education_map = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
job_map = {title: idx for idx, title in enumerate(job_title_options)}

# Prediction on button click
if st.button("🎯 Predict Salary"):
    if experience > (age - 15):
        st.error("❌ Experience can't be greater than Age minus 15.")
    else:
        input_data = np.array([[age, gender_map[gender], education_map[education], job_map[job_title], experience]])
        salary = model.predict(input_data)[0]
        st.success(f"💰 Predicted Salary: **Rs {salary:,.2f}**")

# Model performance section
st.markdown("---")
st.subheader("📊 Model Performance on Test Data")
st.write(f"✅ **Mean Squared Error (MSE):** {mse:.2f}")
st.write(f"✅ **R² Score:** {r2:.4f}")

st.markdown("---")
st.markdown( "<p class='footer-text'>Created by Gayatri </t> from Sahyadri College Of Engineering And Management</p>",
    unsafe_allow_html=True)
