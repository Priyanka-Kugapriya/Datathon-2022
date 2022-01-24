# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 16:41:48 2022

@author: RUVINDI
"""
#Web Development
import streamlit as st

#Numerical Computation
import numpy as np

#Data frames 
import pandas as pd

#Simulate Real time data
import time

#For visualizations
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.express as px

#To load Pickle files
import pickle

import os
#import base64
#from wordcloud import WordCloud, STOPWORD




#Load models for Loan Approval Predictions

#Random Forest classifier
L_RF_pickle_in = open("Models\Cred-RFclassifier.pkl","rb")
L_RF_classifier =pickle.load(L_RF_pickle_in)

#Decision Tree model
L_DT_pickle_in = open("Models\Cred-DTClassifier.pkl","rb")
L_DT_classifier=pickle.load(L_DT_pickle_in)

#KNN model
L_KNN_pickle_in = open("Models\Cred-KNNClassifier.pkl","rb")
L_KNN_classifier=pickle.load(L_KNN_pickle_in)






#Load models for High-Risk Identification

#Random Forest classifier
D_RF_pickle_in = open("Models\Cred-RFclassifier.pkl","rb")
D_RF_classifier =pickle.load(D_RF_pickle_in)

#Decision Tree model
D_DT_pickle_in = open("Models\Cred-DTClassifier.pkl","rb")
D_DT_classifier=pickle.load(D_DT_pickle_in)

#KNN model
D_KNN_pickle_in = open("Models\Cred-KNNClassifier.pkl","rb")
D_KNN_classifier=pickle.load(D_KNN_pickle_in)



# Main Method
def main():
    
    #Title
    st.title("Automatic Loan Approval System 🏦")
    st.markdown("## with Identification of the High-Risk Customers")
    
    #Side Bar
    st.sidebar.title("Automatic Loan Approval System 🏦")
    st.sidebar.markdown("Real Time Bank Data Analysis")
    activities = ["Home", "Loan Approval System", "Identify the High-Risk Customers"]
    choice = st.sidebar.selectbox("Go To", activities)
    
    
    
    #Visualization of the data
    if choice == 'Home':
        #Reading the data from Source
        df = pd.read_csv("Risk_Prediction.csv")
        
        sentiment_count = df['default'].value_counts()
        #st.write(sentiment_count)
        sentiment_count = pd.DataFrame({'default':sentiment_count.index, 'Count':sentiment_count.values})

        st.title("Number Of customers according to Marital Status ")
        fig = px.bar(sentiment_count, x='default' , y='Count',color='Count',height=500)
        st.plotly_chart(fig) 
        
     
     
    
    #Loan Approval System
    if choice == 'Loan Approval System':
        st.info("Predicting whether the Loan can Approve")
        
        #Inputs
        
        loan_amount = st.text_input("Amount of the loan:")
        funded_amnt = st.text_input("Funded Amount:")
        funded_amnt_inv = st.text_input("Funded Amount Inv:")
        term = st.text_input("Term:")
        int_rate = st.text_input(" Int Rate:")
        installment = st.text_input("Installment Amount:")
        grade = st.text_input("Grade:")
        sub_grade = st.text_input("Sub-Grade:")
        home_ownership = st.text_input("Home ownership:")
        annual_inc = st.text_input("Annual Inc:")
        ##Add the other feild
        
        #Feature list values
        feature_list = [loan_amount,funded_amnt,funded_amnt_inv,term,int_rate,installment,grade,sub_grade,home_ownership,annual_inc,]
        single_sample = np.array(feature_list).reshape(1,-1)
        
        #Selecting the model
        model_choice = st.selectbox("Select Model",["Random Forest Classification","Decision Tree Classifier", "KNN Classifier"])

        
        st.text("")
        
        #Predicting Using the models
	
        if st.button("Predict Outcome"):
            if model_choice == "Random Forest Classification":
                prediction = L_RF_classifier.predict(single_sample)
                pred_prob = L_RF_classifier.predict_proba(single_sample)
            elif model_choice == "Decision Tree Classifier":
                prediction = L_DT_classifier.predict(single_sample)
                pred_prob = L_DT_classifier.predict_proba(single_sample)
            else:
                prediction = L_KNN_classifier.predict(single_sample)
                pred_prob = L_KNN_classifier.predict_proba(single_sample)
                
                
            #Displaying the Predicted Outcome
            if prediction == 'TRUE' :
                st.text("")
                st.warning("This Customer is a High-Risk Customer")
               
                		
            else:
                st.text("")
                st.success("This Customer is not a High-Risk Customer")
                
                
                
                
                

    #Identify the High-Risk Customers
    if choice == 'Identify the High-Risk Customers':
        st.info("Predicting whether the customer is a high-risk customer")
        
        #Inputs
        
        loan_amount = st.text_input("Amount of the loan:")
        duration = st.text_input("Loan Duration:")
        payments = st.text_input("Last Payment amount:")
        order_amount = st.text_input("Orderd Amount:")
        n_inhabitants = st.text_input(" Number of Inhabitants:")
        average_salary = st.text_input("Average Salary of the Customer:")
        entrepreneur_rate = st.text_input("Entrepreneur Rate:")
        Day_between_account_creation_and_loan_application = st.text_input("Day between account creation and loan application:")
        average_unemployment_rate = st.text_input("Average Unemployment Rate:")
        average_crime_rate = st.text_input("Average Crime Rate:")
        
        
        #Feature list values
        feature_list = [loan_amount,duration,payments,order_amount,n_inhabitants,average_salary,entrepreneur_rate,Day_between_account_creation_and_loan_application,average_unemployment_rate,average_crime_rate]
        single_sample = np.array(feature_list).reshape(1,-1)
        
        #Selecting the model
        model_choice = st.selectbox("Select Model",["Random Forest Classification","Decision Tree Classifier", "KNN Classifier"])

        
        st.text("")
        
        #Predicting Using the models
	
        if st.button("Predict Outcome"):
            if model_choice == "Random Forest Classification":
                prediction = D_RF_classifier.predict(single_sample)
                pred_prob = D_RF_classifier.predict_proba(single_sample)
            elif model_choice == "Decision Tree Classifier":
                prediction = D_DT_classifier.predict(single_sample)
                pred_prob = D_DT_classifier.predict_proba(single_sample)
            else:
                prediction = D_KNN_classifier.predict(single_sample)
                pred_prob = D_KNN_classifier.predict_proba(single_sample)
                
                
            #Displaying the Predicted Outcome
            if prediction == 'TRUE' :
                st.text("")
                st.warning("This Customer is a High-Risk Customer")
               
                		
            else:
                st.text("")
                st.success("This Customer is not a High-Risk Customer")
                       
    
                
               
if __name__ == '__main__':
	main()