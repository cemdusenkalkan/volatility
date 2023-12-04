"""
here is for freeing data from errors
missing values etc.
don't know what to add as indicators,
tests will tell me and i will move with the
results, shouldn't keep up with the same data
all the time.
"""


from config import CSV_FILE_PATH, WINDOW_SIZE
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def precompute_future_volatility(dataframe, future_window):
    dataframe['Daily_Return'] = dataframe['Close'].pct_change()
    dataframe['Future_Volatility'] = dataframe['Daily_Return'].rolling(window=future_window).std().shift(-future_window)
    dataframe.dropna(inplace=True)
    return dataframe


df = pd.read_csv(CSV_FILE_PATH)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = precompute_future_volatility(df, future_window=30)


df['Rolling_Mean'] = df['Daily_Return'].rolling(window=WINDOW_SIZE).mean()
df['Rolling_Std'] = df['Daily_Return'].rolling(window=WINDOW_SIZE).std()
df['Rolling_Max'] = df['Close'].rolling(window=WINDOW_SIZE).max()
df['Rollin_Min'] = df['Close'].rolling(window=WINDOW_SIZE).min()
df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()


X = df[['Rolling_Mean', 'Rolling_Std', 'Rolling_Max', 'Rollin_Min', 'EMA_10']]
y = df['Future_Volatility']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
