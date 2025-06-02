"""Unit tests for the sim module."""

import unittest

import pandas as pd
import numpy as np

from vbase_utils.sim import sim


class TestSim(unittest.TestCase):
    """Test cases for the sim module."""

    def setUp(self):
        """Set up test fixtures."""
        self.dates = pd.date_range("2023-01-01", periods=5)
        self.df1 = pd.DataFrame({"A": [1, 2, 3, 4, 5]}, index=self.dates)
        self.df2 = pd.DataFrame({"B": [10, 20, 30, 40, 50]}, index=self.dates)
        self.series = pd.Series([100, 200, 300, 400, 500], index=self.dates, name="C")
        self.sample_data = [self.df1, self.df2, self.series]
        self.time_index = pd.date_range("2023-01-01", periods=5)

    def test_basic_functionality(self):
        """Test basic functionality of the sim function."""

        def callback(data):
            df1, df2, series = data
            return pd.Series(
                [
                    df1["A"].tail(1).values[0],
                    df2["B"].tail(1).values[0],
                    (df1["A"].tail(1) + df2["B"].tail(1) + series.tail(1)).values[0],
                ],
                index=["A", "B", "all"],
            )

        result = sim(self.sample_data, callback, self.time_index)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.time_index))
        pd.testing.assert_index_equal(result.index, self.time_index)
        self.assertIn("all", result.columns)

        # Check first row (should only have first day's data)
        self.assertEqual(result.iloc[0]["all"], 111)  # 1 + 10 + 100

        # Check last row (should have all data)
        self.assertEqual(result.iloc[-1]["all"], 555)  # 5 + 50 + 500

    def test_invalid_index(self):
        """Test that non-DatetimeIndex raises ValueError."""
        df = pd.DataFrame({"A": [1, 2, 3]}, index=[1, 2, 3])
        time_index = pd.date_range("2023-01-01", periods=3)

        # pylint: disable=unused-argument
        def callback(data):
            return pd.Series([1, 2, 3])

        with self.assertRaisesRegex(ValueError, "must have a DatetimeIndex"):
            sim([df], callback, time_index)

    def test_callback_not_series(self):
        """Test that callback returning non-Series raises ValueError."""

        # pylint: disable=unused-argument
        def callback(data):
            return pd.DataFrame({"A": [1, 2, 3]})

        with self.assertRaisesRegex(ValueError, "must return a pandas Series"):
            sim([self.df1], callback, self.time_index)

    def test_callback_exception(self):
        """Test that callback exceptions are properly handled."""

        def callback(data):
            raise ValueError("Test error")

        with self.assertRaisesRegex(ValueError, "Error processing timestamp"):
            sim([self.df1], callback, self.time_index)

    def test_empty_data(self):
        """Test with empty data list."""

        # pylint: disable=unused-argument
        def callback(data):
            return pd.Series([1, 2, 3])

        result = sim([], callback, self.time_index)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.time_index))

    def test_data_masking(self):
        """Test that data is properly masked at each timestamp."""

        def callback(data):
            df1, _, _ = data
            # Return the length of available data at each timestamp
            return pd.Series([len(df1)], index=["data_length"])

        result = sim(self.sample_data, callback, self.time_index)

        # Check that data length increases with each timestamp
        self.assertTrue(result["data_length"].is_monotonic_increasing)
        self.assertEqual(result["data_length"].iloc[0], 1)  # First timestamp
        self.assertEqual(result["data_length"].iloc[-1], 5)  # Last timestamp

    def test_missing_data(self):
        """Test handling of data with missing values."""
        df1 = pd.DataFrame({"A": [1, np.nan, 3, 4, 5]}, index=self.dates)
        df2 = pd.DataFrame({"B": [10, 20, np.nan, 40, 50]}, index=self.dates)

        def callback(data):
            df1, df2 = data
            return pd.Series(df1["A"] + df2["B"], index=["result"])

        result = sim([df1, df2], callback, self.time_index)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.time_index))
        self.assertTrue(
            pd.isna(result.iloc[1]["result"])
        )  # Should be NaN where either input is NaN


if __name__ == "__main__":
    unittest.main()
