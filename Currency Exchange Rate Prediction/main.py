# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_csv('exchange_rate_data.csv', parse_dates=['Date'])
data = data.set_index('Date')

# Sort the DataFrame by the datetime index
data = data.sort_index()

# Filter data from 2006 to April 2024
data = data['2006':'2024-04']

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.values)

# Function to create dataset
def create_dataset(data, time_step):
    X, y = [], []
    if len(data) < time_step:
        print("Dataset length is less than time step. Cannot create sequences.")
        return None, None
    for i in range(len(data) - time_step + 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step - 1, 0])  # Adjusted to take the last value of each sequence
    print("Length of X:", len(X))
    print("Length of y:", len(y))
    return np.array(X), np.array(y)

# Choose the time step
time_step = 100

# Create the dataset
X, y = create_dataset(scaled_data, time_step)

# Check if X and y are not None before attempting to print their shapes
if X is not None and y is not None:
    print("Shape of X before reshaping:", X.shape)
    print("Shape of y:", y.shape)

# Reshape input to be [samples, time steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split the data into train and test
train_size = int(len(X) * 0.67)
test_size = len(X) - train_size
train_X, test_X = X[:train_size], X[train_size:]
train_y, test_y = y[:train_size], y[train_size:]

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(train_X, train_y, epochs=100, batch_size=32)

# Save the trained model
model.save('trained_model')

# Save the training data to a file
np.savez('training_data.npz', train_X=train_X, train_y=train_y, test_X=test_X, test_y=test_y)

# Make predictions for future values
future_predictions = []
last_sequence = scaled_data[-time_step:]
for i in range(12):  # Predicting 12 months into the future
    prediction = model.predict(np.reshape(last_sequence, (1, time_step, 1)))
    future_predictions.append(prediction[0][0])
    last_sequence = np.append(last_sequence[1:], prediction[0])

# Inverse transform the future predictions
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Print future predictions
print("Future predictions:")
print(future_predictions)
