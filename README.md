# Portfolio Optimizer

This command line app takes a text file of stock tickers and uses recent stock data to determine the optimal percentage of each stock for a portfolio. It also calculates the number of stocks you would need to buy at the stocks' current prices to build the portfolio. Stock portfolios are optimized using max sharpe ratio.

## Files Included
- portfolio_optimizer.py - contains the PortfolioOptimizer class that contains the core computations for determining the optimal portfolio using max Sharp Ratio
- main.py - provides a simple use of the PortfolioOptimizer class
- test_portfolio_optimizer.py - set of unit tests for the PortfolioOptimizer class
