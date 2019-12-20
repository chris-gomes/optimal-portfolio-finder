import unittest
import pandas as pd
import numpy as np

from portfolio_optimizer import PortfolioOptimizer

class TestPortfolioOptimizer(unittest.TestCase):
    def setUp(self):
        stock_returns = pd.DataFrame({'stock1':[0.10, -0.5, 0.7], 'stock2':[0.31, 0.8, 0.54]})
        rf_rates = np.array([0.01, 0.12, 0.06])
        self.po = PortfolioOptimizer(stock_returns, rf_rates)

    # test if weights arg is null
    def test_is_valid_weights_null(self):
        self.assertRaisesRegex(
            ValueError,
            'weights cannot be None',
            self.po.is_valid_weights,
            None
        )

    # test if weights arg is not a list
    def test_is_valid_weights_not_list(self):
        self.assertRaisesRegex(
            ValueError,
            'weights must be a list',
            self.po.is_valid_weights,
            4
        )
    
    # test if weights arg is not the same length as the number of stocks
    def test_is_valid_weights_not_same_length(self):
        self.assertRaisesRegex(
            ValueError,
            'weights must be the same length as the number of stocks',
            self.po.is_valid_weights,
            [0.3, 0.4, 0.3]
        )

    # tests if the weights do not add up to 1
    def test_is_valid_weights_sum_not_1(self):
        self.assertRaisesRegex(
            ValueError,
            'weights must sum up to 1',
            self.po.is_valid_weights,
            [0.3, 0.4]
        )

    # test a valid weight arg
    def test_is_valid_weights_is_true(self):
        self.assertTrue(self.po.is_valid_weights([0.3, 0.7]))

    # test if an equally weighted portfolio return is caluated correctly
    def test_port_ret_equal_weights(self):
        self.assertAlmostEqual(self.po.port_ret([0.5, 0.5]), 3.9)

    # test if an unequally weighted portfolio return is caluated correctly
    def test_port_ret_unequal_weights(self):
        self.assertAlmostEqual(self.po.port_ret([0.7, 0.3]), 2.82)

    # test if an equally weighted portfolio sd is caluated correctly
    def test_port_sd_equal_weights(self):
        self.assertAlmostEqual(self.po.port_sd([0.5, 0.5]), 0.8901124)

    # test if an unequally weighted portfolio sd is caluated correctly
    def test_port_sd_unequal_weights(self):
        self.assertAlmostEqual(self.po.port_sd([0.7, 0.3]), 1.3373810)

if __name__ == '__main__':
    unittest.main()