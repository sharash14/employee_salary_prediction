import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load trained model and columns
with open("salary_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

# Title
st.title("Income Predictor")
st.markdown("### Predict whether income is >50K or <=50K")

# Input form
with st.form("input_form"):
    age = st.slider("Age", 17, 75, 30)
    workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 
                                           'Local-gov', 'State-gov', 'Without-pay', 'Never-worked', 'Others'])
    marital_status = st.selectbox("Marital Status", ['Married-civ-spouse', 'Divorced', 'Never-married', 
                                                     'Separated', 'Widowed', 'Married-spouse-absent'])
    occupation = st.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial',
                                             'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical',
                                             'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv',
                                             'Armed-Forces', 'NotListed'])
    relationship = st.selectbox("Relationship", ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
    race = st.selectbox("Race", ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
    gender = st.radio("Gender", ['Male', 'Female'])
    capital_gain = st.number_input("Capital Gain", 0, 50000, 0)
    educational_num = st.slider("Educational Number (5-16)", 5, 16, 10)
    hours_per_week = st.slider("Hours per Week", 1, 99, 40)
    native_country = st.selectbox("Native Country", ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'India', 
                                                     'England', 'China', 'Others'])
    submitted = st.form_submit_button("Predict")

# Preprocessing: Manual encoding (same format used while training)
if submitted:
    # Categorical Dictionaries (based on Label Encoders used)
    workclass_map = {'Private': 3, 'Self-emp-not-inc':6, 'Self-emp-inc':5, 'Federal-gov':1, 'Local-gov':2, 'State-gov':7,
                     'Without-pay':8, 'Never-worked':0, 'Others':4}
    marital_map = {'Married-civ-spouse':1, 'Divorced':0, 'Never-married':2, 'Separated':3, 'Widowed':5, 'Married-spouse-absent':4}
    occupation_map = {'Tech-support':12, 'Craft-repair':1, 'Other-service':7, 'Sales':10, 'Exec-managerial':2,
                      'Prof-specialty':8, 'Handlers-cleaners':4, 'Machine-op-inspct':5, 'Adm-clerical':0,
                      'Farming-fishing':3, 'Transport-moving':13, 'Priv-house-serv':9, 'Protective-serv':11,
                      'Armed-Forces':14, 'NotListed':6}
    relationship_map = {'Wife':5, 'Own-child':2, 'Husband':1, 'Not-in-family':3, 'Other-relative':4, 'Unmarried':0}
    race_map = {'White':4, 'Asian-Pac-Islander':1, 'Amer-Indian-Eskimo':0, 'Other':2, 'Black':3}
    gender_map = {'Male':1, 'Female':0}
    country_map = {'United-States':38, 'Mexico':20, 'Philippines':29, 'Germany':10, 'Canada':4, 'India':13, 
                   'England':7, 'China':5, 'Others':30}

    # Encode user input
    input_data = pd.DataFrame([[
        age,
        workclass_map.get(workclass, 4),
        fnlwgt := 100000,  # you can keep it default as not used in your model
        education_num := educational_num,
        marital_map.get(marital_status, 2),
        occupation_map.get(occupation, 6),
        relationship_map.get(relationship, 3),
        race_map.get(race, 4),
        gender_map.get(gender, 1),
        capital_gain,
        capital_loss := 0,
        hours_per_week,
        country_map.get(native_country, 30)
    ]], columns=model_columns)

    # Predict
    prediction = model.predict(input_data)[0]
    label = ">50K" if prediction == 1 else "<=50K"
    st.success(f"Predicted Income: {label}")
