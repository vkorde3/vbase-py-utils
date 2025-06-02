"""Time-based simulation module for processing time series data."""

from typing import Callable, List, Union
import logging

import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def sim(
    data: List[Union[pd.DataFrame, pd.Series]],
    callback: Callable[[List[Union[pd.DataFrame, pd.Series]]], pd.Series],
    time_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Simulate processing of time series data using a callback function.

    This function simulates processing of time series data by:
    1. Iterating through each timestamp in the provided time index
    2. For each timestamp, masking all data after that timestamp
    3. Calling the provided callback function with the masked data
    4. Collecting the results into a single DataFrame

    Args:
        data: List of pandas DataFrames and/or Series containing time series data.
            All objects must have a DatetimeIndex.
        callback: Function that processes the masked data and returns a Series.
            The function should accept a list of DataFrames/Series and return a Series.
        time_index: DatetimeIndex specifying the simulation timestamps.
            The function will process data up to each timestamp in this index.

    Returns:
        DataFrame containing the results of the callback function for each timestamp.
        The index will match the provided time_index.

    Raises:
        ValueError: If any input data object doesn't have a DatetimeIndex.
        ValueError: If the callback function doesn't return a Series.
        ValueError: If the callback function raises an exception.
    """
    # Validate input data
    for i, obj in enumerate(data):
        if not isinstance(obj.index, pd.DatetimeIndex):
            raise ValueError(
                f"Data object at index {i} must have a DatetimeIndex, "
                f"got {type(obj.index)}"
            )

    # Initialize results DataFrame
    l_df_results: List[pd.DataFrame] = []

    # Process each timestamp
    for timestamp in time_index:
        try:
            # Mask data for current timestamp
            masked_data = [obj[obj.index <= timestamp] for obj in data]

            # Call callback function
            result = callback(masked_data)

            # Validate callback result
            if not isinstance(result, pd.Series):
                raise ValueError(
                    f"Callback must return a pandas Series, got {type(result)}"
                )

            # Turn this Series into a DataFrame with the timestamp time index.
            df_result = pd.DataFrame([result], index=[timestamp])
            l_df_results.append(df_result)

        except Exception as e:
            logger.error(
                "Error processing timestamp %s: %s",
                timestamp,
                str(e),
                exc_info=True,
            )
            raise ValueError(f"Error processing timestamp {timestamp}: {str(e)}") from e

    # Combine all results into a single DataFrame
    return pd.concat(l_df_results)
