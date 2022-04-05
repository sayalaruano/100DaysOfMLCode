import streamlit as st

st.header('Creating a button')

if st.button('Say hello'):
     st.write('Why hello there')
else:
     st.write('Goodbye')