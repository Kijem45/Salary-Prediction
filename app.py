import streamlit as st 
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_model():
    with open("saved_salary_prediction.pk1", "rb") as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map

def clean_experience(x):
    if x == "More than 50 years":
        return 50
    if x == "Less than 1 year":
        return 0.5
    return float(x)

def clean_education(x):
    if pd.isna(x):
        return x
    if "Bachelor degree" in x:
        return "Bachelor degree"
    if "Master degree" in x:
        return "Master degree"
    if "Professional degree" in x or "Associate degree" in x:
        return "Post grad"
    return "Less than a Bachelor"

@st.cache_data
def load_data():
    df = pd.read_csv("survey_results_public.csv")
    df = df[['Country', 'EdLevel', 'YearsCodePro', 'Employment', 'ConvertedCompYearly']]
    df = df.rename({'ConvertedCompYearly': 'Salary'}, axis=1)
    df = df[df['Salary'].notnull()]
    df = df.dropna()
    df = df[df['Employment'] == 'Employed, full-time']
    df = df.drop('Employment', axis=1)
    country_map = shorten_categories(df.Country.value_counts(), 400)
    df['Country'] = df['Country'].map(country_map)
    df = df[df['Salary'] <= 250000]
    df = df[df['Salary'] >= 10000]
    df = df[df['Country'] != 'Other']
    df['YearsCodePro'] = df['YearsCodePro'].apply(clean_experience)
    df["EdLevel"] = df["EdLevel"].apply(clean_education)
    return df

df = load_data()

def explore():
    st.title("Software Developer Salaries(2023) Analysis")
    #st.write("#### Developer Survey 2023")

    # Visualization
    # data = df["Country"].value_counts()
    # fig1, ax1 = plt.subplots()
    # ax1.pie(data, labels=data.index, autopct="%1.1f%%", shadow=True, startangle=90)
    # ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.

    # st.write("#### Number of Data from Different Countries")
    # st.pyplot(fig1)

    st.write("###### Mean Salary Based on Country")
    data = df.groupby(["Country"])["Salary"].mean().sort_values(ascending=True)
    st.bar_chart(data)

    st.write("###### Mean Salary Based on Experience")
    data = df.groupby(["YearsCodePro"])["Salary"].mean().sort_values(ascending=True)
    st.line_chart(data)

    st.write("###### Data from different countries")
    data = df["Country"].value_counts()
    st.bar_chart(data)



    



    st.write("##### ")

def predict():
    st.subheader("Software Developer Salary Prediction")

    Countries = (
        "United States of America",
        "Germany", "United Kingdom of Great Britain and Northern Ireland",
        "Canada", "India", "France", "Netherlands",
        "Australia", "Brazil", "Spain", "Sweden",
        "Italy", "Poland", "Switzerland", "Denmark",
        "Norway", "Israel"
    )

    education_levels = (
        "Less than a Bachelor",
        "Bachelor degree",
        "Master degree",
        "Professional degree"
    )


    country = st.selectbox("Country", Countries)
    education = st.selectbox("Education Level", education_levels)
    experience = st.slider("Years of Experience", 0, 50, 3)

    ok = st.button("Calculate Salary")
    if ok:
        x = np.array([[country, education, experience]])
        x[:, 0] = le_country.transform(x[:, 0])
        x[:, 1] = le_education.transform(x[:, 1])
        x = x.astype(float)

        salary = regressor.predict(x)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")

session = st.sidebar.selectbox("Explore Or Predict", ("Predict", "Explore"))

if session == "Predict":
    predict()
else:
    explore()
