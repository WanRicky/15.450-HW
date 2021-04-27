import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima.model import ARIMA

# creator: Ruiqin Wan, 2021.4.27
# github: https://github.com/WanRicky/15.450-HW/blob/main/Assignment5.py
# To run the codes, just change the q3_a/b/c/d/e in line 132


def get_data(file_path, sheet_name):
    df = pd.DataFrame(pd.read_excel(file_path, sheet_name=sheet_name, index_col=0))
    return df


def q3_a():
    print("begin")
    df = get_data("data/HW5_WMT.xlsx", "HW5_WMT")
    df.index = pd.to_datetime(df.index, format='%Y%m%d')
    print(df)
    df['first_difference'] = np.log(df['WMT']) - np.log(df['WMT']).shift(1)
    df['season_difference'] = np.log(df['WMT']) - np.log(df['WMT']).shift(4)
    df['first_difference'].plot()
    df['season_difference'].plot()
    plt.show()


def q3_b():
    print("begin")
    df = get_data("data/HW5_WMT.xlsx", "HW5_WMT")
    df.index = pd.to_datetime(df.index, format='%Y%m%d')
    df['first_difference'] = np.log(df['WMT']) - np.log(df['WMT']).shift(1)
    df['season_difference'] = np.log(df['WMT']) - np.log(df['WMT']).shift(4)
    df = df.head(df.index.get_loc('2016-03-31'))
    print(df)
    ARIMA_model = ARIMA(np.log(df['WMT']), order=(0, 1, 1)).fit()  # p=0, d=1, q=1
    print(ARIMA_model.summary())
    ARIMA_model.predict().plot()
    np.log(df['WMT']).plot()
    plt.show()


def q3_c():
    print("begin")
    df = get_data("data/HW5_WMT.xlsx", "HW5_WMT")
    df.index = pd.to_datetime(df.index, format='%Y%m%d')
    df['first_difference'] = np.log(df['WMT']) - np.log(df['WMT']).shift(1)
    df['season_difference'] = np.log(df['WMT']) - np.log(df['WMT']).shift(4)
    df = df.head(df.index.get_loc('2016-03-31'))
    print(df)
    airline_model = ARIMA(np.log(df['WMT']), order=(0, 1, 1), seasonal_order=(0, 1, 1, 4)).fit()
    print(airline_model.summary())
    airline_model.predict().plot()
    np.log(df['WMT']).plot()
    plt.show()


def q3_d():
    print("begin")
    df = get_data("data/HW5_WMT.xlsx", "HW5_WMT")
    df.index = pd.to_datetime(df.index, format='%Y%m%d')
    df['first_difference'] = np.log(df['WMT']) - np.log(df['WMT']).shift(1)
    df['season_difference'] = np.log(df['WMT']) - np.log(df['WMT']).shift(4)
    df_test = df.tail(len(df.index) - df.index.get_loc('2016-03-31'))
    df_test = df_test.head(df_test.index.get_loc('2020-03-31'))
    df_p = df.head(df.index.get_loc('2016-03-31'))
    print(df_test)
    rst_arima_list = []
    rst_airline_list = []
    i = 1
    for index in df_test.index:
        ARIMA_model = ARIMA(np.log(df_p['WMT']), order=(0, 1, 1)).fit()  # p=0, d=1, q=1
        airline_model = ARIMA(np.log(df_p['WMT']), order=(0, 1, 1), seasonal_order=(0, 1, 1, 4)).fit()
        rst_arima_list.append(ARIMA_model.forecast()[0])
        rst_airline_list.append(airline_model.forecast()[0])
        df_p = df.head(df.index.get_loc('2016-03-31') + i)
        i += 1

    plt.plot(df_test.index, rst_arima_list, label='ARIMA Model')
    plt.plot(df_test.index, rst_airline_list, label='AIRLINE Model')
    np.log(df_test['WMT']).plot(label='Reality')
    plt.legend()
    plt.show()


def q3_e():
    print("begin")
    df = get_data("data/HW5_WMT.xlsx", "HW5_WMT")
    df.index = pd.to_datetime(df.index, format='%Y%m%d')
    df['first_difference'] = np.log(df['WMT']) - np.log(df['WMT']).shift(1)
    df['season_difference'] = np.log(df['WMT']) - np.log(df['WMT']).shift(4)
    df_test = df.tail(len(df.index) - df.index.get_loc('2016-03-31'))
    df_test = df_test.head(df_test.index.get_loc('2020-03-31'))
    df_p = df.head(df.index.get_loc('2016-03-31'))
    print(df_test)
    rst_arima_list = []
    rst_airline_list = []
    i = 1
    for index in df_test.index:
        ARIMA_model = ARIMA(np.log(df_p['WMT']), order=(0, 1, 1)).fit()  # p=0, d=1, q=1
        airline_model = ARIMA(np.log(df_p['WMT']), order=(0, 1, 1), seasonal_order=(0, 1, 1, 4)).fit()
        rst_arima_list.append(ARIMA_model.forecast()[0])
        rst_airline_list.append(airline_model.forecast()[0])
        df_p = df.head(df.index.get_loc('2016-03-31') + i)
        i += 1

    rst_arima_error_list = []
    rst_airline_error_list = []
    for j in range(0, len(df_test), 1):
        rst_arima_error_list.append(np.log(df_test.iloc[j].at['WMT']) - rst_arima_list[j])
        rst_airline_error_list.append(np.log(df_test.iloc[j].at['WMT']) - rst_airline_list[j])

    plt.plot(df_test.index, rst_arima_error_list, label='ARIMA ERROR')
    plt.plot(df_test.index, rst_airline_error_list, label='AIRLINE ERROR')
    plt.legend()
    plt.show()

    arima_mse = 0
    airline_mse = 0
    for j in range(0, len(df_test), 1):
        arima_mse += rst_arima_error_list[j] * rst_arima_error_list[j]
        airline_mse += rst_airline_error_list[j] * rst_airline_error_list[j]
    arima_mse = arima_mse / len(df_test)
    airline_mse = airline_mse / len(df_test)
    print("arima_mse = ", arima_mse)
    print("airline_mse = ", airline_mse)


q3_e()
