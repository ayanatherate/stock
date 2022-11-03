###################################### library imports ############################################################################################

import streamlit as st
import numpy as np
import pandas as pd
from nsepy import get_history as gh
import time
import datetime as dt
from datetime import date
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

#################### setting page configurations #####################################################################################################

st.set_page_config(page_title='STOCK PRICE ESTIMATOR', page_icon="https://cdn-icons-png.flaticon.com/512/4449/4449895.png", layout="centered", initial_sidebar_state="auto", menu_items=None)
hide_streamlit_style2= '''
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.css-1rs6os {visibility: hidden;}
.css-17ziqus {visibility: hidden;}
</style>
'''
st.markdown(hide_streamlit_style2, unsafe_allow_html=True) 

page_bg_img = '''
<style>
.stApp {
background-image: url("https://www.semarchy.com/wp-content/uploads/2021/12/header-blue3.png");
background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

st.image('https://cdn-icons-png.flaticon.com/512/4946/4946378.png',width=100)

################## input from user #######################################################################################################################

st.title(f'Track and forecast NSE stock prices')
stock_to_show=st.text_input('Enter a stock code to start: ')

####################### data processing/feature engineering###################################################################################################################

start = dt.date(2013,11,1)
end = date.today()

if stock_to_show: 
  stk_data = gh(symbol=stock_to_show,start=start,end=end)
else:
    st.stop()

if len(stk_data)>0:
  st.subheader(f'Historical Price Data of {stock_to_show}')
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=stk_data.index,y=stk_data['Open'],name='Open', mode="lines"))
  fig.add_trace(go.Scatter(x=stk_data.index,y=stk_data['Close'],name='Close', mode="lines"))
  st.plotly_chart(fig,use_container_width=True)
else:
  st.write('Error! If you have entered the wrong Stock Code Entered. Please cross-check and enter the correct NSE stock code! Alternatively, if the server is down, kindly wait and try again after some time.')
  st.stop()

start = dt.date(2013,11,1)
end = dt.date(2021,10,9)

stk_data = gh(symbol=stock_to_show,start=start,end=end)
stk_data['Date'] = stk_data.index

data2 = pd.DataFrame(columns = ['Date', 'Open', 'High', 'Low', 'Close'])
data2['Date'] = stk_data['Date']
data2['Open'] = stk_data['Open']
data2['High'] = stk_data['High']
data2['Low'] = stk_data['Low']
data2['Close'] = stk_data['Close']

x_train=data2['Close']
range_data=max(x_train)
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    """Generates dataset windows

    Args:
      series (array of float) - contains the values of the time series
      window_size (int) - the number of time steps to average
      batch_size (int) - the batch size
      shuffle_buffer(int) - buffer size to use for the shuffle method

    Returns:
      dataset (TF Dataset) - TF Dataset containing time windows
    """
  
    # Generate a TF Dataset from the series values
    dataset = tf.data.Dataset.from_tensor_slices(series)
    
    # Window the data but only take those with the specified size
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    
    # Flatten the windows by putting its elements in a single batch
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))

    # Create tuples with features and labels 
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))

    # Shuffle the windows
    dataset = dataset.shuffle(shuffle_buffer)
    
    # Create batches of windows
    dataset = dataset.batch(batch_size).prefetch(1)
    
    return dataset

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    """Generates dataset windows

    Args:
      series (array of float) - contains the values of the time series
      window_size (int) - the number of time steps to average
      batch_size (int) - the batch size
      shuffle_buffer(int) - buffer size to use for the shuffle method

    Returns:
      dataset (TF Dataset) - TF Dataset containing time windows
    """
  
    # Generate a TF Dataset from the series values
    dataset = tf.data.Dataset.from_tensor_slices(series)
    
    # Window the data but only take those with the specified size
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    
    # Flatten the windows by putting its elements in a single batch
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))

    # Create tuples with features and labels 
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))

    # Shuffle the windows
    dataset = dataset.shuffle(shuffle_buffer)
    
    # Create batches of windows
    dataset = dataset.batch(batch_size).prefetch(1)
    
    return dataset

window_size = 5
batch_size = 50
shuffle_buffer_size = 1000

train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
#train_set1 = windowed_dataset(train_cl, window_size, batch_size, shuffle_buffer_size)


testdataframe= gh(symbol=stock_to_show,start=dt.date(2021,10,10),end=date.today())
testdataframe['Date'] = testdataframe.index
testdata = pd.DataFrame(columns = ['Date', 'Open', 'High', 'Low', 'Close'])
testdata['Date'] = testdataframe['Date']
testdata['Open'] = testdataframe['Open']
testdata['High'] = testdataframe['High']
testdata['Low'] = testdataframe['Low']
testdata['Close'] = testdataframe['Close']

x_test=testdata['Close']
val_set = windowed_dataset(x_test, window_size, batch_size, shuffle_buffer_size)

################################### model building #################################################################################################

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv1D(filters=256, kernel_size=3,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=[window_size, 1]),
  tf.keras.layers.LSTM(128, return_sequences=True),
  tf.keras.layers.LSTM(64,return_sequences=False),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(16, activation='relu'),
  tf.keras.layers.Dense(8, activation='relu'),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * range_data)
])


lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20))
# simple early stopping

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=100)


mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

# Initialize the optimizer
optimizer = tf.keras.optimizers.SGD(momentum=0.9)

# Set the training parameters
model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer,metrics='mae')
tf.keras.backend.clear_session()
init_weights = model.get_weights()
model.set_weights(init_weights)

########################### model training and output ##############################################################################################################
yesno=st.text_input('Do you want to forecast future stock prices (Y/N)?')
if yesno=='Y':
    st.write('Starting Model Training. Kindly wait while the Deep Learning model completes its training process.')
    st.write('Model: 1D CNN-LSTM')
    st.write('Model Architecture: Conv1D(f=256,k=3)-->LSTM(128)-->LSTM(64)-->Dropout(0.2)-->Dense(16)--->Dense(8)--->Dense(1)-->Lambda')
    st.write('Estimated training on: 7 years of stock price data')
    st.write('Estimated Training time: 1-2 minutes')
    
    history = model.fit(train_set,validation_data=[val_set],epochs=500,callbacks=[es,lr_schedule,mc])
    saved_model = load_model('best_model.h5')
    loss=history.history['val_loss']
    lowest_loss=min(history.history['val_loss'])
    for i in range(len(loss)):
      if loss[i]==lowest_loss:
        stripe=i
        break

    epochs=[i for i in range(len(loss))][:i]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs,y=history.history['loss'][:i],name='Loss', mode="lines"))
    fig.add_trace(go.Scatter(x=epochs,y=history.history['val_loss'][:i],name='Val_loss', mode="lines"))
    st.plotly_chart(fig, use_container_width=True)

    val_mae=min(history.history['val_mae'])
    st.write('Loss curve for the model')
    st.write('Callbacks applied: Early Stopping, Model Checkpoint, Learning Rate Scheduler')
    st.write('Only the best model with the minimum Validation loss was selected.')
    st.write(f'Validation MAE: {val_mae}')
    print(f"prediction dataset day start: {testdata['Date'][len(testdata)-5]}")
    prediction_start_array=testdata['Close'][len(testdata)-5:].values
    prediction_start_array=list(prediction_start_array)

    

    i=0
    year_end_forecasts=[]
    while i!=8:
      dataset = tf.data.Dataset.from_tensor_slices(prediction_start_array)
      dataset = dataset.window(window_size, shift=1, drop_remainder=True)
      dataset = dataset.flat_map(lambda w: w.batch(window_size))
      dataset = dataset.batch(batch_size).prefetch(1)
      forecast = saved_model.predict(dataset)
      result=forecast.squeeze()
      print(result)
      year_end_forecasts.append(result)
      prediction_start_array.remove(prediction_start_array[0])
      prediction_start_array.append(result)
  
      i+=1
    
    
    average_forward_5=np.average(year_end_forecasts)
    
    if average_forward_5>testdata['Close'].values[-1]:
      st.caption('Trend:')
      st.image('https://cdn-icons-png.flaticon.com/512/2601/2601574.png')
    else:
      st.caption('Trend:')
      st.image('https://cdn-icons-png.flaticon.com/512/3121/3121773.png')

    st.write(f'The estimated average Closing price of the stock, 5 days forward is Rs. {average_forward_5}')
else:
  st.stop()

###########################################################################################################################################################
    
















