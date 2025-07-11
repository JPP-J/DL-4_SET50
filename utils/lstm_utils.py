import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Example data
data = df[['Close']].copy()

# Normalize data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Create sequences
lookback = 10
X, y = [], []
for i in range(lookback, len(scaled_data)):
    X.append(scaled_data[i-lookback:i, 0])  # 10 previous days
    y.append(scaled_data[i, 0])             # today's close
X, y = np.array(X), np.array(y)

# Reshape for LSTM (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(units=64, return_sequences=False, input_shape=(lookback, 1)))
model.add(Dense(units=1))  # output layer

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=20, batch_size=32, validation_split=0.1)


predicted_scaled = model.predict(X)
predicted = scaler.inverse_transform(predicted_scaled.reshape(-1, 1))
actual = scaler.inverse_transform(y.reshape(-1, 1))


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(actual, label='Actual Close Price')
plt.plot(predicted, label='Predicted Close Price')
plt.legend()
plt.title("LSTM Stock Price Prediction")
plt.grid(alpha=0.3)
plt.show()
