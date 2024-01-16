import streamlit as st
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns

with open('model.pkl','rb') as file:
    model = pkl.load(file)

with open('preproc.pkl','rb') as f:
    pre = pkl.load(f)

st.header("Your Loan is ready, get approval instantly!!!")

def features():
    st.sidebar.header('User Input Features')

    checkin_acc = st.sidebar.selectbox('Checking Amount', ['A11', 'A12', 'A13', 'A14'])
    savings_acc = st.sidebar.selectbox('Savings Amount', ['A61', 'A62', 'A63', 'A64', 'A65'])
    credit_history = st.sidebar.selectbox('Credit History', ['A30', 'A31', 'A32', 'A33', 'A34'])
    present_emp_since = st.sidebar.selectbox('Employment status', ['A71', 'A72', 'A73', 'A74', 'A75'])
    personal_status= st.sidebar.selectbox('Personal Loan status', ['A91', 'A92', 'A93', 'A94'])
    amount= st.sidebar.slider('Amount', 1, 100000, 100)
    duration= st.sidebar.slider('Duration', 1, 100, 1)
    job = st.sidebar.selectbox('Job status', ['A171', 'A172', 'A173', 'A174'])
    inst_rate = st.sidebar.slider('Installment Rate',1,4,1)
    residing_since = st.sidebar.slider('Residing Since',1,4,1)
    age = st.sidebar.slider('Age',18,80,1)
    num_credits=st.sidebar.slider('Numeric Credits',1,4,1)
    inst_plans=st.sidebar.selectbox('Installment Plans',['A141','A142','A143'])

    data = {'checkin_acc': checkin_acc, 'savings_acc': savings_acc, 'credit_history': credit_history, 'present_emp_since': present_emp_since,
            'personal_status': personal_status, 'amount': amount, 'duration': duration,'job':job,'inst_rate':inst_rate,
            'residing_since':residing_since,'age':age,'num_credits':num_credits,'inst_plans':inst_plans}

    input_vars = pd.DataFrame(data, index=[0])
    return input_vars

def pred_status(input_data):
    df_tf = pre.transform(input_data)
    pred_status = model.predict(df_tf)[0]
    return pred_status 

df = features()

pred_status = pred_status(df)

st.subheader('Predicted Loan Status')
st.write(f'{pred_status}')