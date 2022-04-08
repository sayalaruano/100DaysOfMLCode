# Imports
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Function to load data of Sars-CoV2 variants in Europe
@st.cache
def load_data():
    df_covid = pd.read_csv("data.csv")

    # Add common names of variants 
    df_covid.loc[(df_covid["variant"] =="B.1.1.529"), "variant"] = "B.1.1.529 - Omicron"
    df_covid.loc[(df_covid["variant"] =="B.1.351"), "variant"] = "B.1.351 - Beta"
    df_covid.loc[(df_covid["variant"] =="B.1.617.2"), "variant"] = "B.1.617.2 - Delta"
    df_covid.loc[(df_covid["variant"] =="P.1"), "variant"] = "P.1 - Gamma"
    df_covid.loc[(df_covid["variant"] =="B.1.1.7"), "variant"] = "B.1.1.7 - Alpha"
    df_covid.loc[(df_covid["variant"] =="B.1.525"), "variant"] = "B.1.525 - Eta"
    df_covid.loc[(df_covid["variant"] =="B.1.617.1"), "variant"] = "B.1.617.1 - Kappa"
    df_covid.loc[(df_covid["variant"] =="B.1.621"), "variant"] = "B.1.621 - Mu"
    df_covid.loc[(df_covid["variant"] =="C.37"), "variant"] = "C.37 - Lambda"
    df_covid.loc[(df_covid["variant"] =="B.1.427/B.1.429"), "variant"] = "B.1.427/B.1.429 - Epsilon"
    df_covid.loc[(df_covid["variant"] =="UNK"), "variant"] = "Unknown"

    return df_covid

# Load data
df = load_data()

# Filter data of Germany
df_covid_ger = df[df["country"] == "Germany"]

# Create donut plots of Sars-CoV2 variant distribution in european countries
donut_plots= {}

for j in df["year_week"].unique():
    df_temp = df_covid_ger[(df_covid_ger["year_week"] ==j) & (df_covid_ger["source"] =="GISAID")]
    plot_temp = go.Figure(data=[go.Pie(labels=df_temp["variant"], 
    values=df_temp["percent_variant"], hole=.4)])
    donut_plots[j] = plot_temp
    

# Create bar plots of the number of detections of Sars-CoV2 variants in european countries
bar_plots = {}

for j in df["year_week"].unique():
    df_temp = df_covid_ger[(df_covid_ger["year_week"] ==j) & (df_covid_ger["source"] =="GISAID")]
    plot_temp = px.bar(df_temp, y='number_detections_variant', x='variant',
            text_auto='.2s', labels={
                    "number_detections_variant": "Number of detections",
                    "variant": "Variant"
                })
    plot_temp.update_traces(textfont_size=12, textangle=0, textposition="outside", showlegend=False)
    bar_plots[j] = plot_temp

# Create streamlit app

st.header('Dashboard of weekly reports of SARS-CoV2 variants in Germany from 2020 until now')

st.subheader('by Sebastián Ayala-Ruano ([@sayalaruano](https://twitter.com/sayalaruano))')

st.sidebar.write('Select a date to show the data')

add_sidebar = st.sidebar.selectbox('Year-week:', df["year_week"].unique())

st.sidebar.write('**Note:** If no plots are displayed, it means that there are no data on those weeks.')

st.sidebar.write('The data is available [here](https://www.ecdc.europa.eu/en/publications-data/data-virus-variants-covid-19-eueea)')

st.sidebar.write('If you have comments or suggestions about this work, please DM by [twitter](https://twitter.com/sayalaruano) or [create an issue](https://github.com/sayalaruano/Dashboard_SARS-CoV2_variants_Europe/issues/new) in the GitHub repository of this project.')


for i in df["year_week"].unique():
    if add_sidebar == i:
        st.write(i)
        st.write("Variant distribution")
        donut_plot = donut_plots[i]
        st.plotly_chart(donut_plot)
        st.write("Number of variants")
        bar_plot = bar_plots[i]
        st.plotly_chart(bar_plot)