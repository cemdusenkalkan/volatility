from data_handler import load_data, preprocess_data
from feature_engineering import precompute_future_volatility
from config import CSV_FILE_PATH, WINDOW_SIZE

""""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from feature_engineering import scaler
"""


def main():
    df = load_data(CSV_FILE_PATH)
    df = preprocess_data(df)
    df = precompute_future_volatility(df, future_window=WINDOW_SIZE)

    """
    x = df[['Rolling_Mean', 'Rolling_Std', 'Rolling_Max', 'Rollin_Min', 'EMA_10']]
    y = df['Future_Volatility']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    scaler functions 
    """



