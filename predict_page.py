import streamlit as st
import pickle
import numpy as np


def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

def show_predict_page():
    st.title("Software Developer Salary Prediction")

    st.write("""### We need some information to predict the salary""")


    countries = (
        "United States of America",
        "Germany",
        "United Kingdom of Great Britain and Northern Ireland",
        "India",
        "Canada",
        "France",
        "Brazil",
        "Spain",
        "Netherlands",
        "Australia",
        "Italy",
        "Poland",
        "Sweden",
        "Russian Federation",
        "Switzerland",
    )

    education = (
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",
        "Less than a Bachelors",
    )

    country = st.selectbox("Country",countries)
    education_level = st.selectbox("Education Level",education)
    
    experience = st.slider("Years of Experience", 0,50,2)

    predicted_salary = st.button("Calculate Salary")
    if predicted_salary:
        X = np.array([[country, education_level, experience ]])
        X[:, 0] = le_country.transform(X[:,0])
        X[:, 1] = le_education.transform(X[:,1])
        X = X.astype(float)

        salary = regressor.predict(X)
        st.subheader(f"The estimate salary is ${salary[0]:.2f}")