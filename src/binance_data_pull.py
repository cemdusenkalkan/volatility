import ccxt
import pandas as pd

exchange = ccxt.binance()


def fetch_daily_data(symbol):

    data = exchange.fetch_ohlcv(symbol, timeframe='5m')
    header = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = pd.DataFrame(data, columns=header)
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.drop('Timestamp', axis=1, inplace=True)
    # Set date as index
    df.set_index('Date', inplace=True)
    return df


df = fetch_daily_data('BTC/USDT')

csv_file_path = 'btc_daily_data.csv'
df.to_csv(csv_file_path)

print(f"Data has been saved to {csv_file_path}")
