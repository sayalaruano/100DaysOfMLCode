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
#df_covid_ger = df[df["country"] == "Germany"]

# Create donut plots of Sars-CoV2 variant distribution in european countries
donut_plots_1country = {}
donut_plots_allcountries = {}

for i in df["country"].unique():
    for j in df["year_week"].unique():
        df_temp = df[(df["year_week"] ==i) & (df["source"] =="GISAID")]
        plot_temp = go.Figure(data=[go.Pie(labels=df_temp["variant"], 
        values=df_temp["percent_variant"], hole=.4, title=i)])
        donut_plots_1country[j] = plot_temp
    
    donut_plots_allcountries[i] = donut_plots_1country
    donut_plots_1country = {}

# Create bar plots of the number of detections of Sars-CoV2 variants in european countries
bar_plots_1country = {}
bar_plots_allcountries = {}

for i in df["country"].unique():
    for j in df["year_week"].unique():
        df_temp = df[(df["year_week"] ==j) & (df["source"] =="GISAID")]
        plot_temp = px.bar(df_temp, y='number_detections_variant', x='variant',
                text_auto='.2s', labels={
                        "number_detections_variant": "Number of detections",
                        "variant": "Variant"
                    })
        plot_temp.update_traces(textfont_size=12, textangle=0, textposition="outside", showlegend=False)
        bar_plots_1country[j] = plot_temp
    
    bar_plots_allcountries[i] = bar_plots_1country
    bar_plots_1country = {}

# Create streamlit app

add_sidebar = st.sidebar.selectbox('Country', df["country"].unique())

for i in df["country"].unique():
    if add_sidebar == i:
        donut_plots = donut_plots_allcountries[i]
        bar_plots = bar_plots_allcountries[i]
        for j in df["year_week"].unique():
            date_select = st.selectbox(f'Select a date', df["year_week"].unique())
            st.write(j)
            st.write("Variant distribution")
            donut_plot = donut_plots[j]
            st.plotly_chart(donut_plot)
            st.write("Number of variants")
            bar_plot = bar_plots[j]
            st.plotly_chart(bar_plot)