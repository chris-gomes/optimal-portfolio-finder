import numpy as np

class OptimalPortfolioFinder():
    def __init__(self, returns):
         self.returns = returns

    # Define the portfolio returns function
    def port_ret(self, weights):
        port_ret = np.sum(self.returns.mean()*weights*12)
        return port_ret

    # Define the portfolio standard deviation function
    def port_std(self, weights):
        port_std = np.sqrt(np.dot(weights.T, np.dot(data_initial.cov()*12, weights)))
        return port_std

    # Define the negative Sharpe Ratio function that we will minimize
    def neg_SR(self, weights):
        SR = (port_ret(weights) - rf_rate['rf'].mean()*12)/port_std(weights)
        return (-1)*SR

# find the portfolio with the largest Sharpe ratio