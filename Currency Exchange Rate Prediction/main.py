import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import requests
import io
import logging
import pickle
import json

# Configure logging
logging.basicConfig(filename='currency_forecaster_model.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Function to fetch exchange rate data from Alpha Vantage API
def fetch_exchange_rate(api_key):
    url = f"https://www.alphavantage.co/query?function=FX_DAILY&from_symbol=USD&to_symbol=LKR&apikey={api_key}&datatype=csv"
    response = requests.get(url)
    if response.status_code == 200:
        logging.info("Exchange rate data fetched from Alpha Vantage API.")
        data = pd.read_csv(io.StringIO(response.text))
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data = data.set_index('timestamp')
        return data
    else:
        logging.error("Failed to fetch exchange rate data from the API. Status Code: %d", response.status_code)
        return None

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

# Log data preprocessing steps
logging.info("Data loaded and sorted.")
logging.info("Data filtered from 2006 to April 2024.")
logging.info("Data normalized using MinMaxScaler.")

# Function to create dataset
def create_dataset(data, time_step):
    X, y = [], []
    if len(data) < time_step:
        logging.error("Dataset length is less than time step. Cannot create sequences.")
        return None, None
    for i in range(len(data) - time_step + 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step - 1, 0])  # Adjusted to take the last value of each sequence
    logging.info("Dataset created. Length of X: %d, Length of y: %d", len(X), len(y))
    return np.array(X), np.array(y)

# Choose the time step
time_step = 100

# Create the dataset
X, y = create_dataset(scaled_data, time_step)

# Reshape input to be [samples, time steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split the data into train and test
train_size = int(len(X) * 0.67)
test_size = len(X) - train_size
train_X, test_X = X[:train_size], X[train_size:]
train_y, test_y = y[:train_size], y[train_size:]

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=128, return_sequences=True, input_shape=(time_step, 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=128))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile the model with a custom learning rate
from keras.optimizers import Adam
optimizer = Adam(lr=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Train the model with more epochs and a different batch size
history = model.fit(train_X, train_y, epochs=150, batch_size=64, validation_data=(test_X, test_y), verbose=1)

# Save the trained model
model.save('currency_forecaster_model.h5')
logging.info("Trained model saved to currency_forecaster_model.h5")

# Save the scaler object
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
logging.info("Scaler object saved to scaler.pkl")

# Save training history
with open('training_history.json', 'w') as f:
    json.dump(history.history, f)
logging.info("Training history saved to training_history.json")

# Fetch latest exchange rate data
api_key = "MXTXWLWQM3G26PQW"
latest_exchange_rate_data = fetch_exchange_rate(api_key)

# Print today's currency value through the API and log it
if latest_exchange_rate_data is not None:
    today_value_api = latest_exchange_rate_data.iloc[-1]['close']
    logging.info("Today's currency value (USD to LKR) from Alpha Vantage API: %.2f", today_value_api)
    print("Today's currency value (USD to LKR) from Alpha Vantage API:", today_value_api)
else:
    logging.error("Failed to fetch today's currency value from the API.")
    print("Failed to fetch today's currency value from the API.")

# Make predictions for future values
future_predictions = []
last_sequence = scaled_data[-time_step:]
for i in range(12):  # Predicting 12 months into the future
    prediction = model.predict(np.reshape(last_sequence, (1, time_step, 1)))
    future_predictions.append(prediction[0][0])
    last_sequence = np.append(last_sequence[1:], prediction[0])

# Inverse transform the future predictions
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Compare predicted values with the value from the API and log the comparison
print("Comparison between predicted values and today's value from the API:")
logging.info("Comparison between predicted values and today's value from the API:")
for i, prediction in enumerate(future_predictions):
    difference = prediction - today_value_api
    print(f"Prediction {i+1}: {prediction[0]:.2f} (Difference: {difference[0]:.2f})")
    logging.info("Prediction %d: %.2f (Difference: %.2f)", i+1, prediction[0], difference[0])

from sklearn.metrics import mean_squared_error, mean_absolute_error

# Predictions on training set
train_predictions = model.predict(train_X)
train_mse = mean_squared_error(train_y, train_predictions)
train_mae = mean_absolute_error(train_y, train_predictions)
print("Training MSE:", train_mse)
print("Training MAE:", train_mae)
logging.info("Training MSE: %.4f", train_mse)
logging.info("Training MAE: %.4f", train_mae)

# Predictions on validation set
val_predictions = model.predict(test_X)
val_mse = mean_squared_error(test_y, val_predictions)
val_mae = mean_absolute_error(test_y, val_predictions)
print("Validation MSE:", val_mse)
print("Validation MAE:", val_mae)
logging.info("Validation MSE: %.4f", val_mse)
logging.info("Validation MAE: %.4f", val_mae)
