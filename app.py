import streamlit as st
from skops.io import load
import pandas as pd

st.title("Housing Price Prediction App")

#load the model
model = load('housing_model.skops')

# Input fields for user to enter data
area = st.number_input('Enter Area in sqft', min_value=0)
bedrooms = st.number_input('Enter Number of Bedrooms', min_value=0)
age = st.number_input('Enter Age of the House in Years', min_value=0)

user_data = pd.DataFrame({
    'Area_sqft': [area],
    'Bedrooms': [bedrooms],
    'Age_years': [age]
})



# Display the predicted price
st.subheader("Predicted Price of the House:")

if st.button("Predict Price"):
    predicted_price = model.predict(user_data)
    st.write(f"${predicted_price[0]:,.2f}")
