from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr
key = "1c8bf9f7f1496a6160903ca865b799f49ce9f463"


start = '2010-01-01'
end = '2021-12-6'

st.title("Stock market prediction")

user_input = st.text_input("Enter Stock Ticker", "AAPL")
df = pdr.DataReader(user_input, 'yahoo', start, end)

# Describe
st.subheader("Data from 2010 to 2021")
st.write(df.describe())

st.subheader("Closing Price vs Time chart")
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader("Closing Price vs Time chart with 100 mean average")
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'r')
plt.plot(df.Close)
st.pyplot(fig)


st.subheader(
    "Closing Price vs Time chart with 100 mean average and 200 mean average")
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close)
st.pyplot(fig)

# Training n testing data
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.75)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.75):])

print(data_training.shape)
print(data_testing.shape)

scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# Model
model = load_model('LSTM_model.h5')

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []
time_step = 100

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-time_step:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# making prediction
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor


# predicted graph
st.subheader("Predicted stock price: ")
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
