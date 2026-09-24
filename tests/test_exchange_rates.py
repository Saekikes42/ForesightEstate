import unittest

import pandas as pd

from src.exchange_rates import convert_sale_prices


class ExchangeRateTests(unittest.TestCase):
    def setUp(self):
        self.rates = pd.DataFrame([
            {"rate_date": "2019-01-04", "currency": "EUR", "try_per_unit": 6.0},
            {"rate_date": "2019-01-07", "currency": "EUR", "try_per_unit": 7.0},
            {"rate_date": "2019-01-04", "currency": "GBP", "try_per_unit": 7.2},
            {"rate_date": "2019-01-04", "currency": "USD", "try_per_unit": 5.0},
        ])
        self.data = pd.DataFrame([
            {"listing_type": 1, "price": 100, "price_currency": "EUR", "start_date": "1/6/19"},
            {"listing_type": 1, "price": 200, "price_currency": "TRY", "start_date": "1/6/19"},
            {"listing_type": 2, "price": 100, "price_currency": "EUR", "start_date": "1/6/19"},
        ])

    def test_uses_prior_rate_and_preserves_try(self):
        result = convert_sale_prices(self.data, self.rates)
        self.assertEqual(len(result), 2)
        self.assertEqual(result.loc[0, "price"], 600)
        self.assertEqual(result.loc[0, "rate_date"], pd.Timestamp("2019-01-04"))
        self.assertEqual(result.loc[1, "price"], 200)
        self.assertTrue(pd.isna(result.loc[1, "rate_date"]))
        self.assertEqual(self.data.loc[0, "price"], 100)

    def test_missing_prior_rate_raises(self):
        data = self.data.iloc[[0]].copy()
        data.loc[0, "start_date"] = "12/20/18"
        with self.assertRaises(ValueError):
            convert_sale_prices(data, self.rates)

    def test_bad_currency_raises(self):
        data = self.data.iloc[[0]].copy()
        data.loc[0, "price_currency"] = "CHF"
        with self.assertRaises(ValueError):
            convert_sale_prices(data, self.rates)


if __name__ == "__main__":
    unittest.main()
