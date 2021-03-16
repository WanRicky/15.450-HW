import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# creator: Ruiqin Wan, 2021.3.15
# github: https://github.com/WanRicky/15.450-HW/blob/main/Assignment2.py
# To run the codes, just change the path in line 16 and line 21


def get_data(file_path, sheet_name):
    df = pd.DataFrame(pd.read_excel(file_path, sheet_name=sheet_name, index_col=0))
    return df


df_temp = get_data("data/HW2_F-F_Research_Data_Factors.xlsx", "q4_b")
df_temp.index = pd.to_datetime(df_temp.index)
df_ff = df_temp.tail(len(df_temp.index) - df_temp.index.get_loc('1971-01-01'))
df_ff = df_ff.head(df_ff.index.get_loc('2021-01-01'))

df_temp = get_data("data/30_Industry_Portfolios.xlsx", "HW2_q4")
df_temp.index = pd.to_datetime(df_temp.index, format='%Y%m')
df_idsty = df_temp.tail(len(df_temp.index) - df_temp.index.get_loc('1971-01-01'))
df_idsty = df_idsty.head(df_idsty.index.get_loc('2021-01-01'))

def q1():
    print("q4_b begin")

    alpha_hat_list = []
    beta_hat_list = []
    for idsty in df_idsty.columns:
        ar_x = list(df_ff["Mkt-RF"])
        ar_y = list(df_idsty[idsty])
        ar_x_c = sm.add_constant(ar_x)
        fit = sm.OLS(ar_y, ar_x_c).fit()
        alpha_hat_list.append(fit.params[0])
        beta_hat_list.append(fit.params[1])
    df_rgrs = pd.DataFrame(columns=[], index=df_idsty.columns)
    df_rgrs['alpha_hat'] = alpha_hat_list
    df_rgrs['beta_hat'] = beta_hat_list
    df_rgrs = df_rgrs.sort_values(by=['beta_hat'])


    for idsty in df_rgrs.index[0:5]:
        plt.title(idsty)
        plt.plot(df_ff['Mkt-RF'],
                 df_rgrs.loc[idsty]['alpha_hat'] + df_rgrs.loc[idsty]['beta_hat'] * df_ff['Mkt-RF'])
        plt.plot(df_ff['Mkt-RF'], df_idsty[idsty], '*')
        # plt.show()
    # print(df_rgrs.index[-5:])
    for idsty in df_rgrs.index[-5:]:
        plt.title(idsty)
        plt.plot(df_ff['Mkt-RF'],
                 df_rgrs.loc[idsty]['alpha_hat'] + df_rgrs.loc[idsty]['beta_hat'] * df_ff['Mkt-RF'])
        plt.plot(df_ff['Mkt-RF'], df_idsty[idsty], '*')
        # plt.show()


# q1()


def q2():
    print("q4_c begin")

    N = 5 * 12
    alpha_hat_list = []
    beta_hat_list = []
    defensive_ret_list = []
    cyclical_ret_list = []
    defensive_raw_ret_list = []
    cyclical_raw_ret_list = []
    mkt_ret_list = []
    mkt_raw_ret_list = []
    ret_list = []
    for i in range(df_idsty.index.get_loc('1981-01-01'), len(df_idsty.index)):
        interval = df_idsty.iloc[i - N:i]
        rgrs_rst = {}
        for idsty in interval.columns:
            ar_x = list(df_ff.iloc[i - N:i]["Mkt-RF"])
            ar_y = list(interval[idsty])
            ar_x_c = sm.add_constant(ar_x)
            fit = sm.OLS(ar_y, ar_x_c).fit()
            alpha_hat_list.append(fit.params[0])
            beta_hat_list.append(fit.params[1])
            rgrs_rst[idsty] = {'alpha_hat': fit.params[0], 'beta_hat': fit.params[1]}
        df_rgrs = pd.DataFrame.from_dict(rgrs_rst, orient='index').sort_values(by=['beta_hat'])
        defensive_ret_list.append(df_idsty.iloc[i][list(df_rgrs.index[0:5])].mean() - df_ff.iloc[i]['RF'])
        cyclical_ret_list.append(df_idsty.iloc[i][list(df_rgrs.index[-5:])].mean() - df_ff.iloc[i]['RF'])
        defensive_raw_ret_list.append(df_idsty.iloc[i][list(df_rgrs.index[0:5])].mean())
        cyclical_raw_ret_list.append(df_idsty.iloc[i][list(df_rgrs.index[-5:])].mean())
        mkt_ret_list.append(df_ff.iloc[i]['Mkt-RF'])
        mkt_raw_ret_list.append(df_ff.iloc[i]['Mkt-RF'] + df_ff.iloc[i]['RF'])
        ret_list.append(df_ff.iloc[i]['Mkt-RF'] + df_ff.iloc[i]['RF'])
    # print(pd.DataFrame.from_dict(returns, orient='index'))

    print("defensive mean return = ", np.nanmean(defensive_ret_list))
    ar_x = mkt_ret_list
    ar_y = defensive_ret_list
    ar_x_c = sm.add_constant(ar_x)
    fit = sm.OLS(ar_y, ar_x_c).fit()
    defensive_alpha_hat = fit.params[0]
    defensive_beta_hat = fit.params[1]
    print("defensive alpha_hat = ", defensive_alpha_hat)
    print("defensive beta_hat = ", defensive_beta_hat)
    print("defensive Sharpe Ratio = ", np.nanmean(defensive_ret_list) / np.nanstd(defensive_ret_list))
    print("defensive Information Ratio = ", (np.nanmean(defensive_ret_list) - np.nanmean(mkt_ret_list))
          / np.nanstd(list(map(lambda x: x[0]-x[1], zip(defensive_ret_list, mkt_ret_list)))))

    max_drawdown = 1
    md_temp = (pd.Series(defensive_raw_ret_list) / 100 + 1).cumprod()
    for i in range(1, len(md_temp)):
        max_drawdown = (md_temp.iloc[i] / md_temp.iloc[:i].max() - 1) \
            if ((md_temp.iloc[i] / md_temp.iloc[:i].max() - 1) < max_drawdown) else max_drawdown
    print("defensive max_drawdown = ", max_drawdown)


    ar_x = mkt_ret_list
    ar_y = cyclical_ret_list
    ar_x_c = sm.add_constant(ar_x)
    fit = sm.OLS(ar_y, ar_x_c).fit()
    cyclical_alpha_hat = fit.params[0]
    cyclical_beta_hat = fit.params[1]
    print("cyclical mean return = ", np.nanmean(cyclical_ret_list))
    print("cyclical alpha_hat = ", cyclical_alpha_hat)
    print("cyclical beta_hat = ", cyclical_beta_hat)
    print("cyclical Sharpe Ratio = ", np.nanmean(cyclical_ret_list) / np.nanstd(cyclical_ret_list))
    print("cyclical Information Ratio = ", (np.nanmean(cyclical_ret_list) - np.nanmean(mkt_ret_list))
          / np.nanstd(list(map(lambda x: x[0] - x[1], zip(cyclical_ret_list, mkt_ret_list)))))

    max_drawdown = 1
    md_temp = (pd.Series(cyclical_raw_ret_list) / 100 + 1).cumprod()
    for i in range(1, len(md_temp)):
        max_drawdown = (md_temp.iloc[i] / md_temp.iloc[:i].max() - 1) \
            if ((md_temp.iloc[i] / md_temp.iloc[:i].max() - 1) < max_drawdown) else max_drawdown
    print("cyclical max_drawdown = ", max_drawdown)

    df_ff_index = df_ff.tail(len(df_ff.index) - df_ff.index.get_loc('1981-01-01'))
    df_rst = pd.DataFrame(columns=[], index=df_ff_index.index)
    df_rst['defensive_raw_ret'] = defensive_raw_ret_list
    df_rst['cyclical_raw_ret'] = cyclical_raw_ret_list
    df_rst['mkt_ret'] = mkt_raw_ret_list

    rst = (df_rst / 100 + 1).cumprod()[['defensive_raw_ret', 'cyclical_raw_ret', 'mkt_ret']]
    plt.plot(rst.index, rst['defensive_raw_ret'], label='defensive_raw_ret')  # blue
    plt.plot(rst.index, rst['cyclical_raw_ret'], label='cyclical_raw_ret')  # orange
    plt.plot(rst.index, rst['mkt_ret'], label='mkt_ret')  # green
    plt.show()


q2()
