import logging
import pandas as pd
from config import CSV_FILE_PATH, MODEL_SAVE_PATH, RANDOM_SEED, WINDOW_SIZE
from data_handler import load_data
from feature_engineering import precompute_future_volatility
from model import VolatilityModel
from evaluation import evaluate_model_performance, plot_predictions_vs_actuals, plot_error_distribution
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(data):
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
    data.dropna(subset=['Close'], inplace=True)
    return data


def main():
    logging.basicConfig(filename='volatility_model.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("Starting volatility modeling process")

    logging.info("Loading data")
    data = load_data(CSV_FILE_PATH)
    if data is None:
        logging.error("Data loading failed")
        return

    data = preprocess_data(data)
    logging.info("Applying feature engineering")
    data = precompute_future_volatility(data, future_window=30)

    # Drop NaN values after each operation
    data.dropna(inplace=True)

    data['rolling_mean'] = data['Daily_Return'].rolling(window=WINDOW_SIZE).mean()
    data['rolling_std'] = data['Daily_Return'].rolling(window=WINDOW_SIZE).std()
    data['rolling_max'] = data['Close'].rolling(window=WINDOW_SIZE).max()
    data['rolling_min'] = data['Close'].rolling(window=WINDOW_SIZE).min()
    data['ema_10'] = data['Close'].ewm(span=10, adjust=False).mean()

    data.dropna(inplace=True)

    required_columns = ['rolling_mean', 'rolling_std', 'rolling_max', 'rolling_min', 'ema_10']
    X = data[required_columns]
    y = data['Future_Volatility']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = VolatilityModel()
    logging.info("Training the model")
    model.train(X_train_scaled, y_train)

    logging.info("Evaluating the model")
    predictions = model.predict(X_test_scaled)
    evaluation_results = evaluate_model_performance(y_test, predictions)
    logging.info(f"Evaluation Results: {evaluation_results}")

    logging.info("Visualizing results")
    plot_predictions_vs_actuals(y_test, predictions)
    plot_error_distribution(y_test, predictions)

    model.save(MODEL_SAVE_PATH)
    logging.info(f"Model saved to {MODEL_SAVE_PATH}")

    logging.info("Volatility modeling process completed")


if __name__ == "__main__":
    main()
