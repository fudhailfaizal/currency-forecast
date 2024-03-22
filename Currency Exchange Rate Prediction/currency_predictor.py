from main import load_data, preprocess_data, train_model, make_prediction  # Import necessary functions from main.py

def main():
    print("Welcome to Currency Exchange Rate Prediction")

    # Load and preprocess data, train model
    X, y = load_data()
    X_processed, y_processed = preprocess_data(X, y)
    model = train_model(X_processed, y_processed)

    while True:
        print("\nSelect a time frame:")
        print("1. Tomorrow")
        print("2. Next Week")
        print("3. Next Month")
        print("4. Next Year")
        print("5. Exit")

        choice = input("Enter your choice (1-5): ")

        if choice == "1":
            time_frame = "Tomorrow"
        elif choice == "2":
            time_frame = "Next Week"
        elif choice == "3":
            time_frame = "Next Month"
        elif choice == "4":
            time_frame = "Next Year"
        elif choice == "5":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")
            continue

        predicted_rates = make_prediction(model, time_frame)
        print(f"\nPredicted Exchange Rates for {time_frame}: {predicted_rates}")

if __name__ == "__main__":
    main()
