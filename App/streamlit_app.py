
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

# Load the trained model and PCA transformer
@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(PCA_PATH, 'rb') as pca_file:
        pca = pickle.load(pca_file)
    return model, pca


MODEL_PATH = "../models/rf_model.pkl"
PCA_PATH = "../models/pca_transformer.pkl"


# Streamlit UI
st.title('Fraud Transaction Prediction App')

st.write("""
Provide transaction details below to predict if it's **Fraudulent** or **Legitimate**.
""")

# Input form
type_options = ['CASH_OUT', 'TRANSFER', 'CASH_IN', 'DEBIT', 'PAYMENT']
type_selected = st.selectbox('Transaction Type:', type_options)
amount = st.number_input('Transaction Amount:', min_value=0.0, format="%.2f")
oldbalanceOrg = st.number_input('Original Balance (Sender):', min_value=0.0, format="%.2f")
newbalanceOrig = st.number_input('New Balance (Sender):', min_value=0.0, format="%.2f")
oldbalanceDest = st.number_input('Original Balance (Receiver):', min_value=0.0, format="%.2f")
newbalanceDest = st.number_input('New Balance (Receiver):', min_value=0.0, format="%.2f")

if st.button('Predict'):
    balance_diff_orig = oldbalanceOrg - newbalanceOrig
    balance_diff_dest = newbalanceDest - oldbalanceDest
    type_encoded = LabelEncoder().fit(type_options).transform([type_selected])[0]

    # Create DataFrame for model
    input_data = pd.DataFrame({
        'step': [1],
        'amount': [amount],
        'oldbalanceOrg': [oldbalanceOrg],
        'newbalanceOrig': [newbalanceOrig],
        'oldbalanceDest': [oldbalanceDest],
        'newbalanceDest': [newbalanceDest],
        'balance_diff_orig': [balance_diff_orig],
        'balance_diff_dest': [balance_diff_dest],
        'type_encoded': [type_encoded]
    })

    # PCA Transformation
    input_pca = pca.transform(input_data)

    # Prediction
    prediction = model.predict(input_pca)[0]
    prediction_prob = model.predict_proba(input_pca)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Transaction is predicted as **FRAUDULENT** with {prediction_prob*100:.2f}% probability!")
    else:
        st.success(f"✅ Transaction is predicted as **LEGITIMATE** with {100 - prediction_prob*100:.2f}% probability!")
