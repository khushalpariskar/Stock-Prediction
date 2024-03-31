import yfinance as yf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import Ridge
import random
import matplotlib.pyplot as plt
import numpy as np
import mplcursors


# Function to fetch historical stock data
def fetch_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

# Function to create features from historical stock data
def create_features(data):
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    return data.dropna()

# Function to split data into features and target
def split_data(data):
    X = data[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_50', 'SMA_200']]
    y = data['Close']
    return X, y

# Function to train a linear regression model
def train_model(X_train, y_train):
    model = Ridge(alpha=1.0)  # You can adjust the alpha parameter for regularization strength
    model.fit(X_train, y_train)
    return model

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = np.mean((predictions - y_test) ** 2)
    return mse

# Function to visualize actual vs predicted prices
# Function to visualize actual vs predicted prices with a subset of data points
def visualize_results(actual, predicted, sample_size=500):
    sample_indices = random.sample(range(len(actual)), min(sample_size, len(actual)))
    actual_prices = actual.values[sample_indices]
    predicted_prices = predicted[sample_indices]
    indices = np.arange(len(actual_prices))

    difference = actual_prices - predicted_prices

    # Create a figure with subplots for actual and predicted prices
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot actual prices
    axs[0].plot(indices, actual_prices, marker='', color='blue', label='Actual')
    axs[0].set_ylabel('Stock Price')
    axs[0].set_title('Actual and Predicted Stock Prices')
    axs[0].legend()

    # Plot predicted prices
    axs[1].plot(indices, predicted_prices, marker='', color='red', label='Predicted')
    axs[1].set_ylabel('Stock Price')
    axs[1].set_xlabel('Time')
    axs[1].legend()

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()

    # Create a separate figure for the bar graph of differences
    plt.figure(figsize=(12, 6))

    min_diff = abs(min(difference))
    barplot = plt.bar(indices, difference, color='green', alpha=0.7)

    # Set y-axis range for the bar graph
    plt.ylim(0, min_diff)

    # Plot difference between actual and predicted prices
    plt.bar(indices, difference, color='green', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Difference')
    plt.title('Difference between Actual and Predicted Stock Prices')
    mplcursors.cursor(barplot, hover=True)

    # Show the plot
    plt.show()

# Main function
def main():
    symbol_list = ['AAPL', 'AMZN', 'MSFT', 'GOOGL']  # Add more symbols as needed

    # Prompt user to select a symbol
    print("Choose a symbol from the following list:")
    for i, symbol in enumerate(symbol_list, 1):
        print(f"{i}. {symbol}")

    # Prompt user for symbol index (with input validation)
    while True:
        try:
            selected_index = int(input("Enter the index of the symbol you want to analyze: "))
            if 1 <= selected_index <= len(symbol_list):
                break  # Exit the loop if the input is valid
            else:
                print("Invalid index. Please enter a number within the range.")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

    symbol = symbol_list[selected_index - 1]

    # Prompt user for start and end dates
    start_date = input("Enter the start date in 'YYYY-MM-DD' format: ")
    end_date = input("Enter the end date in 'YYYY-MM-DD' format (or press Enter for today's date): ")
    if not end_date:
        end_date = datetime.today().strftime('%Y-%m-%d')

    # Fetch stock data
    stock_data = yf.download(symbol, start=start_date, end=end_date)

    # Print the fetched stock data
    print(stock_data)

    # Fetch historical stock data
    stock_data = fetch_stock_data(symbol, start_date, end_date)

    # Create features
    processed_data = create_features(stock_data)

    # Split data into features and target
    X, y = split_data(processed_data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    mse = evaluate_model(model, X_test, y_test)
    print("Mean Squared Error:", mse)

    # Make predictions
    predictions = model.predict(X_test)

    # Visualize results
    visualize_results(y_test, predictions)

if __name__ == "__main__":
    main()
