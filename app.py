import streamlit as st
import pandas as pd
import pickle

st.title("Heart Disease Predictor")
tab1, tab2 = st.tabs(['Predict', 'Model Information'])

with tab1:
    # Inputs
    age = st.number_input("Age (years)", min_value=0, max_value=150)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300)
    cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0)
    fasting_bs = st.selectbox("Fasting Blood Sugar", [0, 1])
    resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST", "LVH"])
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=-5.0, max_value=10.0, step=0.1)
    st_slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Up", "Flat", "Down"])

    # Map to numerical values
    sex_map = {"M": 1, "F": 0}
    chest_map = {"ATA": 0, "NAP": 1, "ASY": 2, "TA": 3}
    ecg_map = {"Normal": 0, "ST": 1, "LVH": 2}
    exang_map = {"Y": 1, "N": 0}
    slope_map = {"Up": 0, "Flat": 1, "Down": 2}

    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex_map[sex]],
        'ChestPainType': [chest_map[chest_pain]],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs],
        'RestingECG': [ecg_map[resting_ecg]],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exang_map[exercise_angina]],
        'Oldpeak': [oldpeak],
        'ST_Slope': [slope_map[st_slope]]
    })

    algonames = ['Decision Trees', 'Logistic Regression', 'Random Forest', 'Support Vector Machine', 'Grid Random Forest']
    modelnames = ['DesicionTree.pkl', 'LogisticR.pkl', 'RandomForest.pkl', 'SVM.pkl', 'GridRF.pkl']

    def predict_heart_disease(data):
        predictions = []
        for modelname in modelnames:
            model = pickle.load(open(modelname, 'rb'))
            prediction = model.predict(data)
            predictions.append(prediction)
        return predictions

    if st.button("Submit"):
        st.subheader('Results')
        st.markdown('---------------------------')

        results = predict_heart_disease(input_data)

        for i in range(len(results)):
            st.subheader(algonames[i])
            if results[i][0] == 0:
                st.write("✅ No heart disease detected.")
            else:
                st.write("⚠️ Heart disease detected.")
            st.markdown('------------------------')

with tab2:
    import plotly.express as px
    data = {'Decision Trees': 80.97, 'Logistic Regression': 85.86, 'Random Forest': 84.23, 'Support Vector Machine': 84.22, 'GridRF': 89.75}
    Models = list(data.keys())
    Accuracies = list(data.values())
    df = pd.DataFrame(list(zip(Models,Accuracies)),columns=['Models','Accuracies'])
    fig = px.bar(df,y='Accuracies',x='Models')
    st.plotly_chart(fig)





