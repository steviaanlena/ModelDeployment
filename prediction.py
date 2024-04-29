pip install joblib
pip install --upgrade streamlit


import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the machine learning model and encode
model = joblib.load('best_model.pkl')
gender_encode= joblib.load('gender_encode.pkl')
label_encode_geo=joblib.load('label_encode_geo.pkl')

def main():
    st.title('Churn Model Deployment')

    #Add user input components
    #input one by one
    creditScore=st.number_input("CreditScore", 100, 1000)
    gender=st.radio("gender", ["Male","Female"])
    geography=st.radio("geographical location", ["France","Spain", "Germany", "Others"])
    age=st.number_input("age", 0, 100)
    tenure=st.number_input("tenure", 0,10)
    balance=st.number_input("balance", 0,1000000)
    numOfProducts=st.number_input("number of products", 1,5)
    hasCrCard=st.number_input("have credit card? [0 for no, 1 for yes]", 0,1)
    isActiveMember=st.number_input("an active member? [0 for no, 1 for yes]", 0,1)
    estimatedSalary=st.number_input("salary", 0,1000000)
    
    data = {'CreditScore': float(creditScore), 'Gender': gender, 'Geography': geography,
            'Age': int(age), 'Tenure':int(tenure), 'Balance': float(balance), 
            'NumOfProducts': int(numOfProducts), 'HasCrCard': int(hasCrCard),
            'IsActiveMember': int(isActiveMember), 'EstimatedSalary': float(estimatedSalary)}
    
    df=pd.DataFrame([list(data.values())], columns=['CreditScore', 'Gender', 'Geography',
                                                    'Age', 'Tenure', 'Balance', 'NumOfProducts',
                                                    'HasCrCard', 'IsActiveMember', 'EstimatedSalary'])

    df=df.replace(gender_encode)
    df=df.replace(label_encode_geo)
    
    if st.button('Make Prediction'):
        features=df      
        result = make_prediction(features)
        st.success(f'The prediction is: {result}')

def make_prediction(features):
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()