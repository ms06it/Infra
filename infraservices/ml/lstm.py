import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load data
with open('log/metrics.json') as f:
    raw_data = json.load(f)

# Convert to DataFrame
data = pd.DataFrame([{
    "timestamp": entry["timestamp"],
    **entry["metrics"]
} for entry in raw_data])
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.set_index('timestamp', inplace=True)

# Feature list
features = ['memory_usage', 'disk_io', 'network_latency', 'error_rate']
target = 'memory_usage'

# Scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[features])
scaled_df = pd.DataFrame(scaled_data, columns=features, index=data.index)

# Create sequences
def create_sequences(data, target_index, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len][target_index])
    return np.array(X), np.array(y)

seq_len = 5
target_index = features.index(target)
X, y = create_sequences(scaled_data, target_index, seq_len)

# Model
model = Sequential()
model.add(LSTM(64, input_shape=(seq_len, len(features))))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=15, batch_size=2, verbose=1)

# Predict
predicted = model.predict(X)

# Inverse transform for comparison
predicted_full = np.zeros((len(predicted), len(features)))
predicted_full[:, target_index] = predicted.flatten()
predicted_inverse = scaler.inverse_transform(predicted_full)[:, target_index]

actual_inverse = data[target].values[seq_len:]
timestamps = data.index[seq_len:]

# Detect anomalies (simple threshold-based)
threshold = 0.80  
anomaly_results = []
for i in range(len(predicted_inverse)):
    actual = actual_inverse[i]
    predicted_val = predicted_inverse[i]
    error = abs(actual - predicted_val)
    is_anomaly = error > threshold
    anomaly_results.append({
        "timestamp": str(timestamps[i]),
        "actual_memory_usage": round(actual, 2),
        "predicted_memory_usage": round(predicted_val, 2),
        "error": round(error, 3),
        "anomaly": is_anomaly
    })

# Save as JSON for GNN
with open('log/gnn_input.json', 'w') as f:
    json.dump(anomaly_results, f, indent=2)

print("✅ Anomaly detection completed and saved to log/gnn_input.json")

