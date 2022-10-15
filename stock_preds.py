# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
from nsepy import get_history as gh
import datetime as dt
from datetime import date
import plotly.express as px
import plotly.graph_objects as go

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
st.set_page_config(page_title='Movie Recommendation App', page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)
st.markdown(page_bg_img, unsafe_allow_html=True)

st.title('Track and estimate stock prices of NSE:FEDERALBNK')

start = dt.date(2013,11,1)
end = date.today()

stk_data = gh(symbol='FEDERALBNK',start=start,end=end)
st.subheader(('Historical Price Data of FEDERALBNK'))
fig = go.Figure()
fig.add_trace(go.Scatter(x=stk_data.index,y=stk_data['Open'],name='Open', mode="lines"))
fig.add_trace(go.Scatter(x=stk_data.index,y=stk_data['Close'],name='Close', mode="lines"))

st.plotly_chart(fig, use_container_width=True)


stk_data['Date'] = stk_data.index
data2 = pd.DataFrame(columns = ['Date', 'Open', 'High', 'Low', 'Close'])
data2['Date'] = stk_data['Date']
data2['Open'] = stk_data['Open']
data2['High'] = stk_data['High']
data2['Low'] = stk_data['Low']
data2['Close'] = stk_data['Close']

print(data2)

model=load_model(r'C:\Users\User\Downloads\hh.h5')

print(model.predict())




