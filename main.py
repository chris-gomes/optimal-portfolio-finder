import pandas as pd
import numpy as np
import datetime

from portfolio_optimizer import PortfolioOptimizer
from pandas_datareader import data


def main():
    stocks = pd.read_csv("stocks.txt", sep="\n", header=None)[0].tolist()
    today = datetime.datetime.now()
    five_years_ago = today - datetime.timedelta(weeks=5*52)
    stock_prices = data.DataReader(stocks,
                                   'yahoo',
                                   five_years_ago.strftime("%Y-%m-%d"),
                                   today.strftime("%Y-%m-%d"))['Adj Close']
    stock_returns = pd.DataFrame()
    for ticker in stock_prices:
        stock_returns[ticker] = np.log(1 + stock_prices[ticker].pct_change())
    stock_returns = stock_returns.resample(rule='BM').sum()

    treasury_yields = data.DataReader("^TNX",
                                      "yahoo",
                                      five_years_ago.strftime("%Y-%m-%d"),
                                      today.strftime("%Y-%m-%d"))['Adj Close']
    rf_rates = np.log(1 + treasury_yields.pct_change())
    rf_rates = rf_rates.resample(rule="BM").sum()

    port_opt = PortfolioOptimizer(stock_returns, rf_rates.values)
    port_weights, port_return = port_opt.find_optimal_port()
    
    portfolio = pd.DataFrame({"Stock":port_opt.get_stocks(), "Weights":port_weights})
    print(portfolio)


main()
