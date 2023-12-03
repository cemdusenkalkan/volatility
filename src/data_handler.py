"""""
Hey, main reason for this function is
to load and read datas from csv files
which we will take the price data.

parameters are file_path(string)
this will be the path to csv file.

return will be dataframe.(loaded data)
"""""

import pandas as pd

def load_data(file_path):
    try:
        data = pd.read.csv(file_path)
        print("loading is succesfull.")
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def preprocess_data(data):
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date',inplace=True)
        return data

#Now, this is a skeleton for sure, not handling
#the missing values, normalizing, exceptions etc.

