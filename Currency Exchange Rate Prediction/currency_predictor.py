import tkinter as tk
from tkinter import ttk
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

# Define the time step
time_step = 100


# Function to preprocess input date and make predictions
def predict_currency_rate():
    # Get the date from the input fields
    day = day_entry.get()
    month = month_entry.get()
    year = year_entry.get()

    input_date_str = f"{day}-{month}-{year}"

    try:
        # Convert the input date string to a datetime object
        input_date = datetime.strptime(input_date_str, "%d-%m-%Y")

        # Load the trained model
        model = load_model('trained_model')

        # Define the scaler
        scaler = MinMaxScaler(feature_range=(0, 1))

        # Extend the dataset for future predictions
        future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), end=input_date)
        future_data = pd.DataFrame(index=future_dates, columns=data.columns)
        data_extended = pd.concat([data, future_data])

        # Interpolate missing values in the extended data
        data_extended.interpolate(method='linear', inplace=True)

        # Process the input date and make predictions
        input_index = data_extended.index.get_loc(input_date)
        if input_index >= time_step:
            # Extract the relevant portion of the data for prediction
            input_data = data_extended.iloc[input_index - time_step + 1: input_index + 1]

            # Scale the input data
            scaled_input_data = scaler.fit_transform(input_data)

            # Reshape input data to match the model input shape
            scaled_input_data = np.reshape(scaled_input_data, (1, time_step, 1))

            # Make predictions
            predicted_rate = model.predict(scaled_input_data)

            # Inverse transform the predicted rate
            predicted_rate = scaler.inverse_transform(predicted_rate)

            # Display the predicted currency rate in the result label
            result_label.config(
                text="Predicted currency rate for {} is: {:.2f}".format(input_date_str, predicted_rate[0][0]))

            # Clear previous prediction label
            previous_prediction_label.config(text="")
        else:
            result_label.config(text="Not enough historical data to make prediction for this date.")

    except ValueError:
        result_label.config(text="Invalid date format. Please use DD-MM-YYYY.")


# Function to reset input and result
def reset():
    day_entry.delete(0, tk.END)
    month_entry.delete(0, tk.END)
    year_entry.delete(0, tk.END)
    result_label.config(text="")
    previous_prediction_label.config(text="")


# Function to close the application
def close_app():
    root.destroy()


# Create the main Tkinter window
root = tk.Tk()
root.title("Currency Forecaster")
root.geometry("400x400")  # Set window size

# Add big title
title_label = ttk.Label(root, text="Currency Forecaster", font=("Helvetica", 20, "bold"))
title_label.pack(pady=(20, 10))

# Add subtitle
subtitle_label = ttk.Label(root, text="USD - LKR (historical data from 2006 to 2024)", font=("Helvetica", 12))
subtitle_label.pack()

# Create a frame for input fields
input_frame = ttk.Frame(root)
input_frame.pack(pady=10)

# Day entry field
day_label = ttk.Label(input_frame, text="Day:")
day_label.grid(row=0, column=0, padx=5, pady=5)
day_entry = ttk.Entry(input_frame, width=5)
day_entry.grid(row=0, column=1, padx=5, pady=5)

# Month entry field
month_label = ttk.Label(input_frame, text="Month:")
month_label.grid(row=0, column=2, padx=5, pady=5)
month_entry = ttk.Entry(input_frame, width=5)
month_entry.grid(row=0, column=3, padx=5, pady=5)

# Year entry field
year_label = ttk.Label(input_frame, text="Year:")
year_label.grid(row=0, column=4, padx=5, pady=5)
year_entry = ttk.Entry(input_frame, width=8)
year_entry.grid(row=0, column=5, padx=5, pady=5)

# Add button to trigger prediction
predict_button = ttk.Button(root, text="Predict", command=predict_currency_rate)
predict_button.pack(side=tk.TOP, padx=10, pady=5, fill=tk.X)

# Add button to reset input and result
reset_button = ttk.Button(root, text="Reset", command=reset)
reset_button.pack(side=tk.TOP, padx=10, pady=5, fill=tk.X)

# Add button to view log
log_button = ttk.Button(root, text="View Log")  # Functionality for log button needs to be added
log_button.pack(side=tk.TOP, padx=10, pady=5, fill=tk.X)

# Add close application button
close_button = ttk.Button(root, text="Close Application", command=close_app)
close_button.pack(side=tk.BOTTOM, padx=10, pady=5, fill=tk.X)

# Add label to display result
result_label = ttk.Label(root, text="")
result_label.pack(side=tk.TOP, padx=10, pady=5)

# Add label to display previously predicted currency rate
previous_prediction_label = ttk.Label(root, text="")
previous_prediction_label.pack(side=tk.TOP, padx=10, pady=5)

# Start the Tkinter event loop
root.mainloop()
