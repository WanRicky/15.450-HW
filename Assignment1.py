import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import statsmodels.api as sm
import math
import random

# creator: Ruiqin Wan, 2021.2.27
# note: 1. if you want to run the codes, please change the calling of "get_data" functions in q3_a\b\c\de
# 2. if you want to see the figures, please delete the "#" in front of the line begin with "plt"


def get_data(file_path, sheet_name):
    df = pd.DataFrame(pd.read_excel(file_path, sheet_name=sheet_name, index_col=0))
    return df


def q3_a():
    print("q3_a begin")
    df = get_data("data/HW1_F-F_Research_Data_Factors.xlsx", "q3")
    print(df)
    df.index = pd.to_datetime(df.index)
    # plt.plot(df.index, df['Mkt-RF'])
    # plt.hist(df['Mkt-RF'], color='blue')
    # plt.show()


def q3_b():
    print("q3_b begin")
    df = get_data("data/HW1_F-F_Research_Data_Factors.xlsx", "q3")
    df.index = pd.to_datetime(df.index)
    mkt_rf_mean = df["Mkt-RF"].mean()
    mkt_rf_std = df["Mkt-RF"].std()
    print("mean = ", mkt_rf_mean)
    print("unbiased std = ", mkt_rf_std)
    std_sum = 0
    for i in range(0, len(df["Mkt-RF"]), 1):
        std_sum += (df.iloc[i].at['Mkt-RF'] - mkt_rf_mean) ** 2
    print("biased std = ", math.sqrt(std_sum/len(df["Mkt-RF"])))

    ar_y = list(df["Mkt-RF"])[1:]
    ar_x = list(df["Mkt-RF"])[:-1]
    ar_x_c = sm.add_constant(ar_x)
    model = sm.OLS(ar_y, ar_x_c)
    fit = model.fit()
    a0 = fit.params[0]
    a1 = fit.params[1]
    print("a0 = ", a0)
    print("a1 = ", a1)
    print(fit.summary())
    print("residual: ", fit.resid)
    print("residual_std: ", fit.resid.std())
    print("mse_residual: ", fit.mse_resid)


def q3_c():
    print("q3_c begin")
    df = get_data("data/HW1_F-F_Research_Data_Factors.xlsx", "q3")
    df.index = pd.to_datetime(df.index)

    df_previous = df.head(df.index.get_loc('2021-01-01'))
    mkt_rf_mean = df_previous["Mkt-RF"].mean()
    std_sum = 0
    for i in range(0, len(df_previous["Mkt-RF"]), 1):
        std_sum += (df_previous.iloc[i].at['Mkt-RF'] - mkt_rf_mean) ** 2
    mkt_rf_std = math.sqrt(std_sum / len(df_previous["Mkt-RF"]))
    print("mkt_rf_mean = ", mkt_rf_mean)
    print("mkt_rf_std = ", mkt_rf_std)

    ar_y = list(df_previous["Mkt-RF"])[1:]
    ar_x = list(df_previous["Mkt-RF"])[:-1]
    ar_x_c = sm.add_constant(ar_x)
    model = sm.OLS(ar_y, ar_x_c)
    fit = model.fit()
    a0 = fit.params[0]
    a1 = fit.params[1]
    ar_std = fit.resid.std()
    print("a0 = ", a0)
    print("a1 = ", a1)
    print("ar_std = ", ar_std)

    # IID Model
    iid_list = []
    for i in range(0, 85, 1):
        iid_list.append(random.gauss(mkt_rf_mean, mkt_rf_std))
    df_after = df.tail(len(df.index) - df.index.get_loc('2014-01-01'))
    df_after["Mkt-RF-estimate"] = iid_list
    # plt.plot(df.index[-85:], list(df_after["Mkt-RF"]), 'r')
    # plt.plot(df.index[-85:], iid_list, 'b')
    # plt.show()
    mse_idd = 0
    for i in range(0, 85, 1):
        mse_idd += (list(df_after["Mkt-RF"])[i] - iid_list[i])**2
    mse_idd /= 85
    print("mse_idd = ", mse_idd)


    # OLS Model
    ols_list = []
    df_previous_one_day = df.tail(len(df.index) - df.index.get_loc('2014-01-01') + 1)
    for i in range(0, 85, 1):
        # ols_list.append(a0 + a1 * df_previous_one_day.iloc[i].at["Mkt-RF"] + random.gauss(0, ols_std))
        ols_list.append(a0 + a1 * df_previous_one_day.iloc[i].at["Mkt-RF"])
    # plt.plot(df.index[-85:], list(df_after["Mkt-RF"]), 'r')
    # plt.plot(df.index[-85:], ols_list, 'b')
    # plt.show()
    mse_ols = 0
    for i in range(0, 85, 1):
        mse_ols += (list(df_after["Mkt-RF"])[i] - ols_list[i]) ** 2
    mse_ols /= 85
    print("mse_ols = ", mse_ols)


def q3_e():  # not in use, turn to q3_de
    df = get_data("data/HW1_F-F_Research_Data_Factors.xlsx", "q3")
    df.index = pd.to_datetime(df.index)

    df_previous = df.head(df.index.get_loc('2014-01-01'))
    mkt_rf_mean = df_previous["Mkt-RF"].mean()
    std_sum = 0
    for i in range(0, len(df_previous["Mkt-RF"]), 1):
        std_sum += (df_previous.iloc[i].at['Mkt-RF'] - mkt_rf_mean) ** 2
    mkt_rf_std = math.sqrt(std_sum / len(df_previous["Mkt-RF"]))
    print("mkt_rf_mean = ", mkt_rf_mean)
    print("mkt_rf_std = ", mkt_rf_std)

    ar_y = list(df_previous["Mkt-RF"])[1:]
    ar_x = list(df_previous["Mkt-RF"])[:-1]
    ar_x_c = sm.add_constant(ar_x)
    model = sm.OLS(ar_y, ar_x_c)
    fit = model.fit()
    a0 = fit.params[0]
    a1 = fit.params[1]
    ols_std = fit.resid.std()
    print("a0 = ", a0)
    print("a1 = ", a1)
    print("mse.std = ", ols_std)

    # IID Model
    iid_list = []
    for i in range(0, 85, 1):
        iid_list.append(random.gauss(mkt_rf_mean, mkt_rf_std))
    df_after = df.tail(len(df.index) - df.index.get_loc('2014-01-01'))
    df_after["Mkt-RF-estimate"] = iid_list
    # print(df_after)
    #plt.plot(df.index[-85:], list(df_after["Mkt-RF"]), 'r')
    #plt.plot(df.index[-85:], iid_list, 'b')
    #plt.show()
    mse_idd = 0
    for i in range(0, 85, 1):
        mse_idd += (list(df_after["Mkt-RF"])[i] - iid_list[i])**2
    mse_idd /= 85
    print("iid_2014-01: ", iid_list[0])
    print("mse_idd = ", mse_idd)


    # OLS Model
    ols_list = []
    df_previous_one_day = df.tail(len(df.index) - df.index.get_loc('2014-01-01') + 1)
    print(df_previous_one_day)
    for i in range(0, 85, 1):
        # ols_list.append(a0 + a1 * df_previous_one_day.iloc[i].at["Mkt-RF"] + random.gauss(0, ols_std))
        ols_list.append(a0 + a1 * df_previous_one_day.iloc[i].at["Mkt-RF"])
    #plt.plot(df.index[-85:], list(df_after["Mkt-RF"]), 'r')
    #plt.plot(df.index[-85:], ols_list, 'b')
    #plt.show()
    mse_ols = 0
    for i in range(0, 85, 1):
        mse_ols += (list(df_after["Mkt-RF"])[i] - ols_list[i]) ** 2
    mse_ols /= 85
    print("ols_2014-01: ", ols_list[0])
    print("mse_ols = ", mse_ols)


def q3_de():  # the result of (d) is contained in the first result in (e), so I put them together.
              # the iid_2014-01 and ar_2014-01 is the result of (d)
    print("q3_de begin")
    df = get_data("data/HW1_F-F_Research_Data_Factors.xlsx", "q3")
    df.index = pd.to_datetime(df.index)

    start_num = df.index.get_loc('2014-01-01')
    iid_list = []
    ar_list = []
    for start_num_i in range(0, 85, 1):
        df_previous = df.head(start_num_i + start_num)
        mkt_rf_mean = df_previous["Mkt-RF"].mean()
        std_sum = 0
        for i in range(0, len(df_previous["Mkt-RF"]), 1):
            std_sum += (df_previous.iloc[i].at['Mkt-RF'] - mkt_rf_mean) ** 2
        mkt_rf_std = math.sqrt(std_sum / len(df_previous["Mkt-RF"]))

        ar_y = list(df_previous["Mkt-RF"])[1:]
        ar_x = list(df_previous["Mkt-RF"])[:-1]
        ar_x_c = sm.add_constant(ar_x)
        model = sm.OLS(ar_y, ar_x_c)
        fit = model.fit()
        a0 = fit.params[0]
        a1 = fit.params[1]
        ols_std = fit.resid.std()

        iid_list.append(random.gauss(mkt_rf_mean, mkt_rf_std))

        df_previous_one_day = df.tail(len(df.index) - (start_num + start_num_i) + 1)
        ar_list.append(a0 + a1 * df_previous_one_day.iloc[0].at["Mkt-RF"])

    df_after = df.tail(len(df.index) - df.index.get_loc('2014-01-01'))
    df_after["Mkt-RF-estimate"] = iid_list
    # print(df_after)
    # plt.plot(df.index[-85:], list(df_after["Mkt-RF"]), 'r')
    # plt.plot(df.index[-85:], iid_list, 'b')
    # plt.show()
    mse_idd = 0
    for i in range(0, 85, 1):
        mse_idd += (list(df_after["Mkt-RF"])[i] - iid_list[i]) ** 2
    mse_idd /= 85
    print("iid_2014-01: ", iid_list[0])
    print("mse_idd = ", mse_idd)

    plt.plot(df.index[-85:], list(df_after["Mkt-RF"]), 'r')
    plt.plot(df.index[-85:], ar_list, 'b')
    plt.show()
    mse_ar = 0
    for i in range(0, 85, 1):
        mse_ar += (list(df_after["Mkt-RF"])[i] - ar_list[i]) ** 2
    mse_ar /= 85
    print("ar_2014-01: ", ar_list[0])
    print("mse_ar = ", mse_ar)



q3_a()
q3_b()
q3_c()
q3_de()
