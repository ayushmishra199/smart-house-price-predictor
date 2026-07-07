import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="House Price Predictor",
    layout="wide"
)

with open("house_model.pkl", "rb") as file:
    model = pickle.load(file)

df = pd.read_csv("house.csv")

st.title("🏠 Advanced House Price Prediction")

st.sidebar.header("House Details")

area = st.sidebar.number_input(
    "Area",
    500,
    5000,
    2000
)

bedrooms = st.sidebar.number_input(
    "Bedrooms",
    1,
    10,
    3
)

bathrooms = st.sidebar.number_input(
    "Bathrooms",
    1,
    10,
    2
)

parking = st.sidebar.number_input(
    "Parking",
    0,
    5,
    1
)

age = st.sidebar.number_input(
    "Age",
    0,
    30,
    5
)

location = st.sidebar.number_input(
    "Location Score",
    1,
    10,
    5
)

st.subheader("Dataset Preview")

st.dataframe(df)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Average Price",
    f"₹{df['Price'].mean():,.0f}"
)

col2.metric(
    "Max Price",
    f"₹{df['Price'].max():,.0f}"
)

col3.metric(
    "Total Houses",
    len(df)
)

st.subheader("Area vs Price")

fig, ax = plt.subplots()

ax.scatter(
    df["Area"],
    df["Price"]
)

st.pyplot(fig)

st.subheader("Correlation Heatmap")

fig2, ax2 = plt.subplots(figsize=(8,5))

sns.heatmap(
    df.corr(),
    annot=True,
    ax=ax2
)

st.pyplot(fig2)

if st.button("Predict House Price"):

    prediction = model.predict(
        [[
            area,
            bedrooms,
            bathrooms,
            parking,
            age,
            location
        ]]
    )

    st.success(
        f"Predicted Price : ₹ {prediction[0]:,.0f}"
    )