#%%
#import relevant libraries (visualization, dashboard, data manipulation)
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
# %%
df_covid = pd.read_csv("../../Data/data.csv")

# Add variants' common name 
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

#%%
# Obtain data of Germany 
df_covid_ger = df_covid[df_covid["country"] == "Germany"]

# Obtain data of one week and a single soure 
df_covid_ger_1week = df_covid_ger[(df_covid_ger["year_week"] =="2022-09") & (df_covid_ger["source"] =="GISAID")]
#%%
fig = px.pie(df_covid_ger_1week, values='percent_variant', names='variant', title='Variant Distribution')
fig.update_traces(textposition='inside')
fig.show()
#%%
fig = go.Figure(data=[go.Pie(labels=df_covid_ger_1week["variant"], values=df_covid_ger_1week["percent_variant"], hole=.4)])
#fig.update_traces(textposition="auto")
fig.show()
#%%
fig = px.bar(df_covid_ger_1week, y='number_detections_variant', x='variant', color="variant",
            text_auto='.2s', labels={
                     "number_detections_variant": "Number of detections",
                     "variant": "Variant"
                 })
fig.update_traces(textfont_size=12, textangle=0, textposition="outside", showlegend=False)
fig.show()

#%%
donut_plots = {}

for i in df_covid_ger["year_week"].unique():
    df_temp = df_covid_ger[(df_covid_ger["year_week"] ==i) & (df_covid_ger["source"] =="GISAID")]
    plot_temp = go.Figure(data=[go.Pie(labels=df_temp["variant"], 
    values=df_temp["percent_variant"], hole=.4, title=i)])
    donut_plots[i] = plot_temp
#%%
bar_plots = {}

for i in df_covid_ger["year_week"].unique():
    df_temp = df_covid_ger[(df_covid_ger["year_week"] ==i) & (df_covid_ger["source"] =="GISAID")]
    plot_temp = px.bar(df_temp, y='number_detections_variant', x='variant',
            text_auto='.2s', labels={
                     "number_detections_variant": "Number of detections",
                     "variant": "Variant"
                 })
    plot_temp.update_traces(textfont_size=12, textangle=0, textposition="outside", showlegend=False)
    bar_plots[i] = plot_temp
#%%
bar_plots_1country = {}
bar_plots_allcountries = {}
for i in df_covid["country"].unique():
    for j in df_covid["year_week"].unique():
        df_temp = df_covid[(df_covid["year_week"] ==j) & (df_covid_ger["source"] =="GISAID")]
        plot_temp = px.bar(df_temp, y='number_detections_variant', x='variant',
                text_auto='.2s', labels={
                        "number_detections_variant": "Number of detections",
                        "variant": "Variant"
                    })
        plot_temp.update_traces(textfont_size=12, textangle=0, textposition="outside", showlegend=False)
        bar_plots_1country[j] = plot_temp
    
    bar_plots_allcountries[i] = bar_plots_1country
    bar_plots_1country = {}

#%%


