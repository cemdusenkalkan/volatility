# main.py
import logging
from data_handler import load_data, preprocess_data
from feature_engineering import precompute_future_volatility
from model import VolatilityModel
from evaluation import evaluate_model_performance, plot_predictions_vs_actuals, plot_error_distribution, \
    plot_feature_importances
from utils import init_logging, log
from config import CSV_FILE_PATH, MODEL_SAVE_PATH, RANDOM_SEED
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    init_logging('volatility_model.log')
    log("Starting volatility modeling process")

    log("Loading and preprocessing data")
    data = load_data(CSV_FILE_PATH)
    if data is None:
        log("Data loading failed", level=logging.ERROR)
        return
    data = preprocess_data(data)

    log("Applying feature engineering")
    data = precompute_future_volatility(data, future_window=30)

    x = data[['Rolling_Mean', 'Rolling_Std', 'Rolling_Max', 'Rollin_Min', 'EMA_10']]
    y = data['Future_Volatility']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_SEED)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = VolatilityModel()
    log("Training the model")
    model.train(x_train_scaled, y_train)

    log("Evaluating the model")
    predictions = model.predict(x_test_scaled)
    evaluation_results = evaluate_model_performance(y_test, predictions)
    log(f"Evaluation Results: {evaluation_results}")

    log("Visualizing results")
    plot_feature_importances(model.feature_importances(), x.columns)
    plot_predictions_vs_actuals(y_test, predictions)
    plot_error_distribution(y_test, predictions)

    model.save(MODEL_SAVE_PATH)
    log(f"Model saved to {MODEL_SAVE_PATH}")

    log("Volatility modeling process completed")


if __name__ == "__main__":
    main()
