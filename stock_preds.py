# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
from nsepy import get_history as gh
import datetime as dt
from datetime import date
import plotly.express as px
import plotly.graph_objects as go
import pickle
from tensorflow.keras.models import load_model
#from tensorflow.keras.Model import load_weights
import tensorflow as tf


page_bg_img = '''
<style>
.stApp {
background-image: url("https://i.ibb.co/wJ1YZz0/pexels-lukas-590014.jpg");
background-size: cover;
}
</style>
'''
stock_to_show=st.text_input('Enter a stock code: ')
st.set_page_config(page_title='Movie Recommendation App', page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)
st.markdown(page_bg_img, unsafe_allow_html=True)

st.title('Track and estimate stock prices of NSE:FEDERALBNK')

start = dt.date(2013,11,1)
end = date.today()

stk_data = gh(symbol=stock_to_show,start=start,end=end)
st.subheader(('Historical Price Data of FEDERALBNK'))
fig = go.Figure()
fig.add_trace(go.Scatter(x=stk_data.index,y=stk_data['Open'],name='Open', mode="lines"))
fig.add_trace(go.Scatter(x=stk_data.index,y=stk_data['Close'],name='Close', mode="lines"))

st.plotly_chart(fig, use_container_width=True)


