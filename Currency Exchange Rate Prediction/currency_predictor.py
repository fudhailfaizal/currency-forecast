import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import logging

# Configure logging
logging.basicConfig(filename='currency_forecaster.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Function to log each step
def log_step(step):
    logging.info(step)

# Function to preprocess input date and make predictions
def predict_currency_rate():
    try:
        # Get the date from the input fields
        day = int(day_entry.get())
        month = int(month_entry.get())
        year = int(year_entry.get())

        log_step("Reading input date...")

        if year <= 2024:
            warning_label.config(text="Please enter a year from 2025 and above.", foreground="red", font=("Helvetica", 10, "bold"))
            log_step("Year entered is before 2025.")
            return
        elif year > 2100:
            warning_label.config(text="Year cannot be over 2100.", foreground="red", font=("Helvetica", 10, "bold"))
            log_step("Year entered is after 2100.")
            return
        elif year > 2040:
            warning_label.config(text="Warning: Predictions beyond 2040 are speculative.", foreground="red", font=("Helvetica", 10, "bold"))
            log_step("Predictions requested beyond 2040.")

        if month < 1 or month > 12:
            warning_label.config(text="Invalid month. Please enter a month between 1 and 12.", foreground="red", font=("Helvetica", 10, "bold"))
            log_step("Invalid month entered.")
            return

        if day < 1 or (day > 31 and month in [1, 3, 5, 7, 8, 10, 12]) or (day > 30 and month in [4, 6, 9, 11]):
            warning_label.config(text=f"Invalid day for month {month}. Please enter a valid day.", foreground="red", font=("Helvetica", 10, "bold"))
            log_step("Invalid day entered.")
            return

        if month == 2 and ((year % 4 == 0 and year % 100 != 0) or year % 400 == 0):
            if day > 29:
                warning_label.config(text=f"Invalid day for February {year}. Please enter a valid day.", foreground="red", font=("Helvetica", 10, "bold"))
                log_step("Invalid day entered for February (leap year).")
                return
        elif month == 2:
            if day > 28:
                warning_label.config(text=f"Invalid day for February {year}. Please enter a valid day.", foreground="red", font=("Helvetica", 10, "bold"))
                log_step("Invalid day entered for February (non-leap year).")
                return

        input_date_str = f"{year}-{month:02d}-{day:02d}"

        # Convert the input date string to a datetime object
        input_date = datetime.strptime(input_date_str, "%Y-%m-%d")

        log_step("Input date processed.")

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

            # Normalize the input data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_input_data = scaler.fit_transform(input_data)

            # Reshape input data to match the model input shape
            scaled_input_data = np.reshape(scaled_input_data, (1, time_step, 1))

            # Train the model with the new input data
            train_X = scaled_input_data
            train_y = np.array([data_extended.iloc[input_index]['Exchange Rate']])
            model.fit(train_X, train_y, epochs=10, verbose=0)

            # Make predictions
            predicted_rate = model.predict(scaled_input_data)

            # Inverse transform the predicted rate
            predicted_rate = scaler.inverse_transform(predicted_rate)

            # Display the predicted currency rate in the result label
            result_label.config(
                text="Predicted currency rate for {} is: {:.2f}".format(input_date_str, predicted_rate[0][0]))

            # Clear previous prediction label
            previous_prediction_label.config(text="")
            log_step("Prediction made successfully.")
        else:
            result_label.config(text="Not enough historical data to make prediction for this date.")
            log_step("Insufficient historical data to make prediction.")

    except ValueError:
        result_label.config(text="Invalid date format. Please use DD-MM-YYYY format.")
        log_step("Invalid date format provided.")

# Function to validate the date format
def is_valid_date(day, month, year):
    try:
        datetime(year, month, day)
        return True
    except ValueError:
        return False

# Function to reset input and result
def reset():
    day_entry.delete(0, tk.END)
    month_entry.delete(0, tk.END)
    year_entry.delete(0, tk.END)
    result_label.config(text="")
    previous_prediction_label.config(text="")
    warning_label.config(text="", foreground="red", font=("Helvetica", 10, "bold"))
    log_step("Input and result reset.")

# Function to close the application
def close_app():
    root.destroy()
    log_step("Application closed.")

# Create the main Tkinter window
root = tk.Tk()
root.title("Currency Forecaster")
root.geometry("400x400")  # Set window size

# Add big title
title_label = ttk.Label(root, text="Currency Forecaster", font=("Helvetica", 20, "bold"))
title_label.pack(pady=(20, 10))
log_step("Application started.")

# Add subtitle
subtitle_label = ttk.Label(root, text="USD - LKR (historical data from 2006 to 2024)", font=("Helvetica", 12))
subtitle_label.pack()
log_step("Subtitle added.")

# Create a frame for input fields
input_frame = ttk.Frame(root)
input_frame.pack(pady=10)
log_step("Input frame created.")

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

# Add warning label
warning_label = ttk.Label(root, text="", foreground="red", font=("Helvetica", 10, "bold"))
warning_label.pack(side=tk.BOTTOM, padx=10, pady=5, fill=tk.X)

# Add close application button
close_button = ttk.Button(root, text="Close Application", command=close_app)
close_button.pack(side=tk.BOTTOM, padx=10, pady=5, fill=tk.X)

# Add label to display result
result_label = ttk.Label(root, text="")
result_label.pack(side=tk.TOP, padx=10, pady=5)

# Add label to display previously predicted currency rate
previous_prediction_label = ttk.Label(root, text="")
previous_prediction_label.pack(side=tk.TOP, padx=10, pady=5)

# Load the model and scaler
model = load_model('currency_forecaster_model.h5')
scaler = MinMaxScaler(feature_range=(0, 1))
data = pd.read_csv('exchange_rate_data.csv', parse_dates=['Date'])
data = data.set_index('Date')
data = data.sort_index()

# Define the time step
time_step = 100

# Start the Tkinter event loop
root.mainloop()
