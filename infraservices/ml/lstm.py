import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Generate synthetic data
np.random.seed(42)
timestamps = pd.date_range('2023-01-01', periods=1440, freq='T')
data = pd.DataFrame({
    'timestamp': timestamps,
    'memory_usage': np.random.rand(1440)
})

# Preprocess data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['memory_usage']])
X, y = [], []
for i in range(60, len(scaled_data)):
    X.append(scaled_data[i-60:i])
    y.append(scaled_data[i])
X, y = np.array(X), np.array(y)

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
model.fit(X, y, epochs=1, batch_size=32)

# Save model
model.save('lstm_model.h5')

