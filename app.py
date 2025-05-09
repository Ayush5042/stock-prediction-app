import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
import joblib

st.title('Stock Trend Prediction')
user_input = st.text_input('Enter Stock Ticker (Only MSFT supported)', 'MSFT')

start = '2010-01-01'
end = '2025-12-31'
df = yf.download(user_input, start=start, end=end)

# Show data
st.subheader('Data of the Stock')
st.write(df.describe())

# Visualization
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df['Close'])
st.pyplot(fig)

# Moving Averages
st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()

fig2, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['Close'], label='Closing Price', color='black')
ax.plot(ma100, label='MA100', color='red')
ax.plot(ma200, label='MA200', color='blue')
ax.legend()
st.pyplot(fig2)

# Data preprocessing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):])

# Load scaler and model
scaler = joblib.load('scaler.save')
model = load_model('stock_model.keras')

# Prepare input
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# Predictions
y_predicted = model.predict(x_test)

# Inverse transform
scale_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Plot
st.subheader('Predicted Price vs Original Price')
fig3 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig3)
