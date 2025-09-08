import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load encoders and model
with open('label_encoded_gender.pkl', 'rb') as file:
    label_encoder = pickle.load(file)
with open('onehot_encoded_country.pkl', 'rb') as file:
    onehot_encoder = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
model = load_model('churn_model.h5')

st.title('Customer Churn Prediction')

# Input fields
credit_score = st.number_input('Credit Score', min_value=0)
country = st.selectbox('Country', ['France', 'Spain', 'Germany'])
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.number_input('Age', min_value=0)
tenure = st.number_input('Tenure', min_value=0)
balance = st.number_input('Balance', min_value=0.0)
products_number = st.number_input('Products Number', min_value=1)
credit_card = st.selectbox('Has Credit Card', [0, 1])
active_member = st.selectbox('Active Member', [0, 1])
estimated_salary = st.number_input('Estimated Salary', min_value=0.0)

if st.button('Predict'):
    # Prepare input data
    input_data = {
        'credit_score': credit_score,
        'country': country,
        'gender': gender,
        'age': age,
        'tenure': tenure,
        'balance': balance,
        'products_number': products_number,
        'credit_card': credit_card,
        'active_member': active_member,
        'estimated_salary': estimated_salary
    }
    input_df = pd.DataFrame([input_data])
    input_df['gender'] = label_encoder.transform(input_df['gender'])
    onehot_country = onehot_encoder.transform(input_df[['country']])
    onehot_country_df = pd.DataFrame(onehot_country, columns=onehot_encoder.get_feature_names_out(['country']))
    input_df = pd.concat([input_df.drop('country', axis=1), onehot_country_df], axis=1)
    input_df_scaled = scaler.transform(input_df)
    prediction = model.predict(input_df_scaled)
    if prediction[0][0] > 0.5:
        st.error('The customer is likely to churn.')
    else:
        st.success('The customer is not likely to churn.')
