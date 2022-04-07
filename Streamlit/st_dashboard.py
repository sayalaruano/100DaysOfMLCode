#%%
#import relevant libraries (visualization, dashboard, data manipulation)
import pandas as pd
import numpy as np 
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime
# %%
df_covid = pd.read_csv("data.csv")
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
fig = px.bar(df_covid_ger_1week, y='number_detections_variant', x='variant', text_auto='.2s',
            title="VOI Germany")
fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
fig.show()


#%%
plots = {}

for i in df_covid_ger["year_week"].unique():
    df_temp = df_covid_ger[(df_covid_ger["year_week"] ==i) & (df_covid_ger["source"] =="GISAID")]
    plot_temp = go.Figure(data=[go.Pie(labels=df_temp["variant"], 
    values=df_temp["percent_variant"], hole=.4, title=i)])
    plots[i] = plot_temp
#%%