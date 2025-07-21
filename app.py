import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
import datetime
import datetime

model=tf.keras.models.load_model('model.keras',compile=False)
with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)
st.title("Patient diabetes prediction")
Pregnancies=st.number_input('Number of Pregnancies you had',0,30)
Glucose=st.number_input('Glucose level')
BloodPressure=st.number_input("Blood Pressure Level (enter between 90 to 300 mmHg)",min_value=60,max_value=300)
SkinThickness=st.number_input('Average Skin Thickness',min_value=10)
Insulin=st.number_input('Enter insulin level')
BMI=st.number_input('Enter you BMI')
DiabetesPedigreeFunction=st.number_input('What is you genetic contribution to diabetes',min_value=0.08,max_value=2.42)
Age=st.slider("Enter age",1,100)

input_data=pd.DataFrame({
    'Pregnancies' :[Pregnancies],
    'Glucose':[Glucose],
    'BloodPressure': [BloodPressure],
    'SkinThickness':[SkinThickness],
    'Insulin':[Insulin],
    'BMI':[BMI],
    'DiabetesPedigreeFunction':[DiabetesPedigreeFunction],
    'Age':[Age],
   })
input_data_scaled=scaler.transform(input_data)

prediction=model.predict(input_data_scaled)
prediction
prediction_probability=prediction[0][0]
prediction_probability
if prediction_probability>0.5:
    st.write(f"patient most likely has diabetes")
    st.write(f"chances of having diabetes {prediction_probability}")
else:
    st.write("Patient most likely will not have diabetes")
    st.write(f"chances of having diabete {prediction_probability}")
