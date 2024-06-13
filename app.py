import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import r2_score

# Load the dataset to define X and y
@st.cache
def load_data():
    df = pd.read_csv('led.csv')
    X = df.drop(columns=['Lifeexpectancy'])
    y = df['Lifeexpectancy']
    return X, y

X, y = load_data()

# Load the models
lr_model = joblib.load('lr_model.pkl')
rf_model_important = joblib.load('rf_model_important.pkl')
encoder = joblib.load('encoder.pkl')

# Define the function to predict life expectancy
def predict_life_expectancy(model, input_data, important_features=None):
    if important_features:
        input_data = input_data[important_features]
    return model.predict(input_data)

# Streamlit app
st.title("Life Expectancy Prediction")

# Input fields
country = st.text_input("Country", "Afghanistan")
year = st.number_input("Year", 2015)
status = st.selectbox("Status", ["Developed", "Developing"])
gdp = st.number_input("GDP", 40000)
schooling = st.number_input("Schooling", 15)
income_composition = st.number_input("Income Composition of Resources", 0.8)

# Create input data frame
input_data = pd.DataFrame({
    'Country': [country],
    'Year': [year],
    'Status': [status],
    'GDP': [gdp],
    'Schooling': [schooling],
    'Incomecompositionofresources': [income_composition]
})

# Fill in missing columns with default values
for col in X.columns:
    if col not in input_data:
        input_data[col] = [X[col].mode()[0]] if X[col].dtype == 'object' else [X[col].mean()]

categorical_features = ['Country', 'Status']
input_data[categorical_features] = encoder.transform(input_data[categorical_features])

# Predict using the chosen model
if r2_score(y, rf_model_important.predict(X[important_rf_features])) > r2_score(y, lr_model.predict(X[important_features])):
    prediction = predict_life_expectancy(rf_model_important, input_data, important_rf_features)
    chosen_model = "Random Forest (Important Features)"
else:
    prediction = predict_life_expectancy(lr_model, input_data, important_features)
    chosen_model = "Linear Regression"

st.write(f"The chosen model for deployment is: {chosen_model}")
st.write(f"Predicted Life Expectancy: {prediction[0]}")
