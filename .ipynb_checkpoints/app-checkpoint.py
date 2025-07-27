import streamlit as st
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open("linear_model.pkl", "rb"))

# Mapping dictionaries
education_options = ['High School', 'Bachelor', 'Master', 'PhD']
job_title_options = ['Data Scientist', 'Software Engineer', 'Manager', 'Analyst']
gender_options = ['Male', 'Female', 'Other']

gender_map = {'Male': 0, 'Female': 1, 'Other': 2}
education_map = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
job_map = {title: idx for idx, title in enumerate(job_title_options)}

# Streamlit App
st.set_page_config(page_title="Salary Predictor", page_icon="💰", layout="centered")
st.title("💼 Employee Salary Predictor")
st.markdown("🔍 Estimate annual salary based on personal and professional details.")

# Input fields in two columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("👤 Age", min_value=18, max_value=70, value=25)
    education = st.selectbox("🎓 Education Level", education_options)
    experience = st.number_input("🧪 Years of Experience", min_value=0, max_value=50, value=1)

with col2:
    gender = st.selectbox("⚧️ Gender", gender_options)
    job_title = st.selectbox("💻 Job Title", job_title_options)

st.markdown("---")

# Prediction
if st.button("🚀 Predict Salary"):
    if experience > (age - 15):
        st.error("❌ Invalid input: Experience can't be greater than Age minus 15.")
    else:
        input_data = np.array([[age,
                                gender_map[gender],
                                education_map[education],
                                job_map[job_title],
                                experience]])
        salary = model.predict(input_data)[0]
        st.success(f"💰 Estimated Annual Salary: **${salary:,.2f}**")
        st.balloons()
