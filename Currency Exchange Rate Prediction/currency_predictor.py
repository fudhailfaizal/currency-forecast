import tkinter as tk
from datetime import datetime
import numpy as np
from keras.models import load_model

from main import data, time_step, scaled_data, scaler

# Load the trained model
model = load_model('trained_model')


# Function to preprocess input date and make predictions
def predict_currency_rate():
    # Get the date from the input field
    input_date_str = date_entry.get()

    try:
        # Convert the input date string to a datetime object
        input_date = datetime.strptime(input_date_str, "%Y-%m-%d")

        # Process the input date and make predictions
        input_index = data.index.get_loc(input_date)
        if input_index >= time_step:
            # Extract the relevant portion of the data for prediction
            input_data = scaled_data[input_index - time_step + 1: input_index + 1]

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
        result_label.config(text="Invalid date format. Please use YYYY-MM-DD.")


# Create the main Tkinter window
root = tk.Tk()
root.title("Currency Rate Predictor")

# Add input field for date
date_label = tk.Label(root, text="Enter date (YYYY-MM-DD):")
date_label.pack()
date_entry = tk.Entry(root)
date_entry.pack()

# Add button to trigger prediction
predict_button = tk.Button(root, text="Predict", command=predict_currency_rate)
predict_button.pack()

# Add label to display result
result_label = tk.Label(root, text="")
result_label.pack()

# Start the Tkinter event loop
root.mainloop()
