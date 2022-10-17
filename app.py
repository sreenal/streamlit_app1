import streamlit as st
import pickle
import joblib
import pandas as pd
##################### #
model = joblib.load("model.pkl") 
# model = joblib.load("model.pkl") 
# scale = pickle.load(open("scale.pkl","rb")) 
scale = joblib.load("scale.pkl") 
st.title('Pima Diabetes Detection Using AI...!') 
st.header('The Data Frame used') 
df = pd.read_csv("diabetes.csv") 
st.dataframe(df.head(5)) 
col1,col2 = st.columns(2) 
with col1: 
    pregnancies=st.number_input('Pregnancies',help="Number between 1 to 10") 
    glucose = st.number_input('Glucose') 
    bloodPressure = st.number_input('Blood Pressure') 
    skinThickness = st.number_input('Skin Thickness') 
with col2: 
    insulin = st.number_input('Insulin') 
    bmi =st.number_input('BMI') 
    dpf = st.number_input('DPF') 
    age = st.number_input('Age') 

if st.button('Predict Diabetes'): 
    st.write(f'The pregnancies count is {int(pregnancies)}') 
    st.write(f'The glucose count is {int(glucose)}') 
    st.write(f'The bloodPressure count is {int(bloodPressure)}') 
    st.write(f'The skinThickness count is {int(skinThickness)}') 
    st.write(f'The insulin count is {int(insulin)}') 
    st.write(f'The bmi count is {int(bmi)}') 
    st.write(f'The dpf count is {int(dpf)}') 
    st.write(f'The age count is {int(age)}') 

    rowDF= pd.DataFrame([pd.Series([pregnancies,glucose,bloodPressure,skinThickness,insulin,bmi,dpf,age])]) 
    rowDF_new = pd.DataFrame(scale.transform(rowDF)) 
    st.subheader('The Independent variables given by the user') 
    st.table(rowDF_new) 
    #model prediction 
    prediction= model.predict_proba(rowDF_new) 
    st.subheader('The predicted Probabilities') 
    st.write(prediction) 
    if prediction[0][1] >= 0.5: 
        valPred = round(prediction[0][1],3) 
        #print(f"The Round val {valPred*100}%") #return render_template('result.html',pred=f'You have a chance of having diabetes.\n\nProbability of you being a diabetic is {valPred*100}%.\n\nAdvice : Exercise Regularly') 
        st.warning(f'You have a chance of having diabetes.\n\nProbability of you being a diabetic is {valPred*100}%.\n\nAdvice : Exercise Regularly', icon="⚠️") 
    else: 
        valPred = round(prediction[0][0],3) 
        st.success(f'Congratulations!!!, You are in a Safe Zone.\n\n Probability of you being a non-diabetic is {valPred*100}%.\n\n Advice : Exercise Regularly and maintain like this..!', icon="✅") 
            #return render_template('result.html',pred=f'Congratulations!!!, You are in a 