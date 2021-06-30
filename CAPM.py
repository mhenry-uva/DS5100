#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 10:58:21 2021

@author: mikehenry
"""

# load modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# risk-free Treasury rate
R_f = 0.0175 / 252
# read in the market data
data = pd.read_csv('capm_market_data.csv')
# drop the date
data.drop(["date"], axis = 1, inplace = True)
# computing the daily returns
data["Daily_Returns_SPY"] = (data.spy_adj_close - data.spy_adj_close.shift(1)) / data.spy_adj_close.shift(1)
data["Daily_Returns_AAPL"] = (data.aapl_adj_close - data.aapl_adj_close.shift(1)) / data.aapl_adj_close.shift(1)
data.drop(["spy_adj_close", "aapl_adj_close"], axis = 1, inplace = True)
data.drop(0, inplace = True)
data.head()
# storing the returns as an array
SPY = data.Daily_Returns_SPY.values
AAPL = data.Daily_Returns_AAPL.values
print(SPY[:5])
print(AAPL[:5])
# calculating excess return
SPY_Excess = SPY - R_f
AAPL_Excess = AAPL - R_f
print(SPY_Excess[-5:])
print(AAPL_Excess[-5:])
#plotting it
plot = plt.scatter(SPY_Excess, AAPL_Excess)
beta = np.dot(SPY_Excess.T, SPY_Excess)**-1 * np.dot(SPY_Excess.T, AAPL_Excess)
print(beta)
#beta sensitivity function
def beta_sensitivity(x,y):
    output = []
    for i in range(x.shape[0]):
        x_temp = x.copy()
        y_temp = y.copy()
#         x_temp = np.delete(x_temp, i)
#         y_temp = np.delete(y_temp, i)
        x_temp = np.delete(x_temp, i).reshape(-1,1)
        y_temp = np.delete(y_temp, i).reshape(-1,1)
        beta = np.dot(x_temp.T, x_temp)**-1 * np.dot(x_temp.T, y_temp)
        output.append((i, beta))
    return output
betas = beta_sensitivity(SPY_Excess, AAPL_Excess)
betas[:5]