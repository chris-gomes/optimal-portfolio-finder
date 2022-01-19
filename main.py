import pandas as pd
import numpy as np
import datetime
import sys

from portfolio_optimizer import PortfolioOptimizer
from pandas_datareader import data


def parseCmdLineArgs(argv):
    if (len(argv) != 3):
        raise Exception("Need to provide a file name of new line delimited stock tickers and number of years")
    
    return argv[1], int(argv[2])


def main(argv):
    file_name, number_of_years = parseCmdLineArgs(argv)

    print("Reading in stock returns...")
    stocks = pd.read_csv(file_name, sep="\n", header=None)[0].tolist()
    today = datetime.datetime.now()
    n_years_ago = today - datetime.timedelta(weeks=number_of_years*52)
    stock_prices = data.DataReader(stocks,
                                   'yahoo',
                                   n_years_ago.strftime("%Y-%m-%d"),
                                   today.strftime("%Y-%m-%d"))['Adj Close']
    stock_returns = pd.DataFrame()
    for ticker in stock_prices:
        stock_returns[ticker] = np.log(1 + stock_prices[ticker].pct_change())
    stock_returns = stock_returns.resample(rule='BM').sum()

    print("Reading in risk free rates...")
    treasury_yields = data.DataReader("^TNX",
                                      "yahoo",
                                      n_years_ago.strftime("%Y-%m-%d"),
                                      today.strftime("%Y-%m-%d"))['Adj Close']
    rf_rates = np.log(1 + treasury_yields.pct_change())
    rf_rates = rf_rates.resample(rule="BM").sum()

    print("Optimizing...")
    port_opt = PortfolioOptimizer(stock_returns, rf_rates.values)
    port_weights, port_return = port_opt.find_optimal_port()
    
    portfolio = pd.DataFrame({"Stock":port_opt.get_stocks(), "Weights":port_weights})
    print(portfolio)


main(sys.argv)
