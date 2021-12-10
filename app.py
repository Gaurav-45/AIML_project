from matplotlib import colors
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr
key = "1c8bf9f7f1496a6160903ca865b799f49ce9f463"
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import math

from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

from datetime import date
import yfinance as yf

from sklearn.model_selection import train_test_split

page_bg_img = '''
<style>
.stApp  {
background-image: url("https://www.nasdaq.com/sites/acquia.prod/files/styles/720x400/public/2020/03/16/stocks-iamchamp-adobe.jpg?h=6acbff97&itok=8CjW1T_R");
background-size: cover;
color: rgba(255,255,255);
}
span{
    color: rgba(255,255,255);
}
label{
    color: rgba(255,255,255);
}
.e1wqxbhn1{
    color:rgb(255,255,255);
}
.effi0qh0{
     color:rgb(255,255,255);
}
.e1wbw4rs0{
     color:rgb(0,0,0);
    
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

start = '2010-01-01'
end = date.today().strftime("%Y-%m-%d")

st.title("Stock market prediction")

stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
user_input = st.selectbox('Select dataset for prediction', stocks)
df = pdr.DataReader(user_input, 'yahoo', start, end)

choices = ('LSTM', 'Linear Regression', 'LSTM vs Linear Regression','Forecast')
choice = st.selectbox('Select algorithm for prediction', choices)

# Describe
st.subheader("Data Description")
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

if choice == 'LSTM':
    # LSTM
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

    import numpy
    from sklearn import metrics
    sum=0
    for i in range(0,len(y_test)):
        sum+=abs(y_test[i]-y_predicted[i])

    scaler = scaler.scale_

    scale_factor = 1/scaler[0]
    y_predicted = y_predicted*scale_factor
    y_test = y_test*scale_factor

    # Accuracy

    # predicted graph
    st.subheader("Predicted stock price using LSTM: ")
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(y_test, 'b', label='Original Price')
    plt.plot(y_predicted, 'r', label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)

    
    
    print("Accuracy", (1-(sum/len(y_test))))

elif choice == 'Linear Regression':
    X=df[['High','Low','Open','Volume']]
    Y=df['Close']

    X_train, X_test, Y_train, Y_test=train_test_split(X,Y,shuffle=False)

    regressor=LinearRegression()
    regressor.fit(X_train,Y_train)

    print(regressor.coef_)
    print(regressor.intercept_)

    predicted=regressor.predict(X_test)

    dataframe=pd.DataFrame(Y_test,predicted)

    dfr=pd.DataFrame({'Actual Price:':Y_test,'Predicted Price:':predicted})
    regressor.score(X_test,Y_test)
    
    st.subheader("Predicted stock price using Regression: ")
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(Y_test.to_numpy(),'b',label='Original Price')
    plt.plot(predicted,'r',label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)

    print("Mean Absolute Error: ",metrics.mean_absolute_error(Y_test,predicted))
    print("Mean Squared Error: ",metrics.mean_squared_error(Y_test,predicted))
    print("Root Mean Squared Error: ",math.sqrt(metrics.mean_squared_error(Y_test,predicted)))


elif choice == 'Forecast':
    @st.cache
    def load_data(ticker):
        data = yf.download(ticker, start, end)
        data.reset_index(inplace=True)
        return data

        
    data = load_data(user_input)

    df_train = data[['Date','Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=3*365)
    forecast = m.predict(future)

    # Show and plot forecast
    st.subheader('Forecast data')
    st.write(forecast.tail())
        
    st.subheader(f'Forecast for 3 years')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

else:
    # LSTM
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

    import numpy
    from sklearn import metrics
    sum=0
    for i in range(0,len(y_test)):
        sum+=abs(y_test[i]-y_predicted[i])


    scaler = scaler.scale_

    scale_factor = 1/scaler[0]
    y_predicted = y_predicted*scale_factor
    y_test = y_test*scale_factor


    # predicted graph
    st.subheader("Predicted stock price using LSTM: ")
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(y_test, 'b', label='Original Price')
    plt.plot(y_predicted, 'r', label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)
    
    print("Accuracy", (1-(sum/len(y_test))))
    ### Regression

    X=df[['High','Low','Open','Volume']]
    Y=df['Close']

    X_train, X_test, Y_train, Y_test=train_test_split(X,Y,shuffle=False)

    regressor=LinearRegression()
    regressor.fit(X_train,Y_train)

    print(regressor.coef_)
    print(regressor.intercept_)

    predicted=regressor.predict(X_test)

    dataframe=pd.DataFrame(Y_test,predicted)

    dfr=pd.DataFrame({'Actual Price:':Y_test,'Predicted Price:':predicted})
    regressor.score(X_test,Y_test)

    st.subheader("Predicted stock price using Regression: ")
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(Y_test.to_numpy(),'b',label='Original Price')
    plt.plot(predicted,'r',label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)

    print("Mean Absolute Error: ",metrics.mean_absolute_error(Y_test,predicted))
    print("Mean Squared Error: ",metrics.mean_squared_error(Y_test,predicted))
    print("Root Mean Squared Error: ",math.sqrt(metrics.mean_squared_error(Y_test,predicted)))


# ToDo's
# Accuracy in LSTM
# Regression me mean squary and other errors.  -- Done
# Stock Drop Down ---  Done
# Aman Make PPT Abhi ke abhi
# Doc Linear regression and SS.
# Sarthak Sleep.