import numpy as np
import pandas as pd

from scipy.optimize import minimize


class PortfolioOptimizer():

    # creates optimizer by taking in monthly stock returns and risk free returns
    def __init__(self, stock_returns, rf_rates):
        if not isinstance(stock_returns, pd.DataFrame):
            raise ValueError('stock_returns must be a pandas data frame')
        elif not isinstance(rf_rates, np.ndarray):
            raise Exception('rf_rates must be a numpy array')
        elif rf_rates.size != stock_returns.shape[0]:
            raise Exception(
                'stock_returns and rf_rates must have the same number of rows')
        else:
            self.stock_returns = stock_returns
            self.rf_rates = rf_rates

    # Gets the list of stocks that the optimizer is working with
    def get_stocks(self):
        return self.stock_returns.columns

    # tests if the weights given are valid
    def is_valid_weights(self, weights):
        if not isinstance(weights, np.ndarray):
            raise ValueError('weights must be a list')
        elif weights.size != self.stock_returns.shape[1]:
            raise ValueError(
                'weights must be the same length as the number of stocks')
        else:
            return True

    # Define the portfolio returns function
    def port_ret(self, weights):
        if self.is_valid_weights(weights):
            return np.sum(self.stock_returns.mean() * weights * 12)

    # Define the portfolio standard deviation function
    def port_sd(self, weights):
        if self.is_valid_weights(weights):
            return np.sqrt(np.dot(weights.T, np.dot(self.stock_returns.cov() * 12, weights)))

    # Define the negative Sharpe Ratio function that we will minimize
    def neg_sharpe_ratio(self, weights):
        sharpe_ratio = (self.port_ret(weights) -
                        self.rf_rates.mean() * 12) / self.port_sd(weights)
        return -1 * sharpe_ratio

    # find the portfolio with the largest Sharpe ratio
    def find_optimal_port(self):
        # initialize constraints for weights
        constraints = (
            {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

        # initialize bounds for weights
        bounds = np.array([(0, 1)] * self.stock_returns.shape[1])

        # create an intial guess of equal weights
        init_guess = np.array([1/self.stock_returns.shape[1]]
                              * self.stock_returns.shape[1])

        # Use the SLSQP (Sequential Least Squares Programming) for minimization
        optimal_port = minimize(self.neg_sharpe_ratio, init_guess,
                                method='SLSQP', bounds=bounds, constraints=constraints)

        return [np.around(optimal_port.x, decimals=2).tolist(), round(-optimal_port.fun, 2)]
