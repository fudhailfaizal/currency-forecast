import tkinter as tk
from datetime import datetime
import numpy as np
from keras.models import load_model
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load data
data = pd.read_csv('exchange_rate_data.csv', parse_dates=['Date'])
data = data.set_index('Date')

# Sort the DataFrame by the datetime index
data = data.sort_index()

# Define the time step and scaler
time_step = 100
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.values)

# Load the trained model
model = load_model('trained_model')

# Function to preprocess input date and make predictions
def predict_currency_rate():
    # Get the date from the input field
    input_date_str = date_entry.get()

    try:
        # Convert the input date string to a datetime object
        input_date = datetime.strptime(input_date_str, "%d-%m-%Y")

        # Extend the dataset for future predictions
        future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), end=input_date)
        future_data = pd.DataFrame(index=future_dates, columns=data.columns)
        data_extended = pd.concat([data, future_data])

        # Interpolate missing values in the extended data
        data_extended.interpolate(method='linear', inplace=True)

        # Scale the extended data
        scaled_data_extended = scaler.transform(data_extended.values)

        # Process the input date and make predictions
        input_index = data_extended.index.get_loc(input_date)
        if input_index >= time_step:
            # Extract the relevant portion of the data for prediction
            input_data = scaled_data_extended[input_index - time_step + 1: input_index + 1]

            # Reshape input data to match the model input shape
            input_data = np.reshape(input_data, (1, time_step, 1))

            # Make predictions
            predicted_rate = model.predict(input_data)

            # Inverse transform the predicted rate
            predicted_rate = scaler.inverse_transform(predicted_rate)

            # Display the predicted currency rate in the result label
            result_label.config(
                text="Predicted currency rate for {} is: {:.2f}".format(input_date_str, predicted_rate[0][0]))
        else:
            result_label.config(text="Not enough historical data to make prediction for this date.")

    except ValueError:
        result_label.config(text="Invalid date format. Please use DD-MM-YYYY.")

# Function to reset input and result
def reset():
    date_entry.delete(0, tk.END)  # Clear input date field
    result_label.config(text="")   # Clear result label

# Create the main Tkinter window
root = tk.Tk()
root.title("Currency Rate Predictor")
root.geometry("400x250")  # Set window size

# Add input field for date
date_label = tk.Label(root, text="Enter date (DD-MM-YYYY):")
date_label.pack()
date_entry = tk.Entry(root)
date_entry.pack()

# Add button to trigger prediction
predict_button = tk.Button(root, text="Predict", command=predict_currency_rate)
predict_button.pack()

# Add button to reset input and result
reset_button = tk.Button(root, text="Reset", command=reset)
reset_button.pack()

# Add label to display result
result_label = tk.Label(root, text="")
result_label.pack()

# Start the Tkinter event loop
root.mainloop()
