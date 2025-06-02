"""Robust timeseries regression module"""

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def exponential_weights(
    n: int,
    half_life: float | None = None,
    lambda_: float | None = None,
) -> np.ndarray:
    """Generate exponential decay weights for n time periods.

    Either half_life or lambda_ must be provided.
    If both are provided, lambda_ is used.

    Args:
        n: Number of time periods.
        half_life: Half-life in time units (e.g., days). Must be positive.
        lambda_: Decay factor (e.g., 0.985). Must be between 0 and 1.

    Returns:
        Normalized exponential decay weights as a numpy array.

    Raises:
        ValueError: If neither half_life nor lambda_ is provided.
        ValueError: If half_life is not positive or lambda_ is not between 0 and 1.
    """
    if half_life is None and lambda_ is None:
        raise ValueError("Either half_life or lambda_ must be provided.")
    if half_life is not None and half_life <= 0:
        raise ValueError("half_life must be positive.")
    if lambda_ is not None and not 0 < lambda_ < 1:
        raise ValueError("lambda_ must be between 0 and 1.")

    if lambda_ is None:
        lambda_ = np.exp(np.log(0.5) / half_life)

    weights: np.ndarray = lambda_ ** np.arange(n - 1, -1, -1)
    return weights / np.sum(weights)  # normalize


# The function must take a large number of arguments
# and conseuqntly has a large number of local variables.
# pylint: disable=too-many-arguments, too-many-locals
def robust_betas(
    df_asset_rets: pd.DataFrame,
    df_fact_rets: pd.DataFrame,
    half_life: float | None = None,
    lambda_: float | None = None,
    min_timestamps: int = 10,
) -> pd.DataFrame:
    """Perform robust regression (RLM) with exponential time-weighting.

    Args:
        df_asset_rets: DataFrame of dependent returns with shape (n_timestamps, n_assets).
        df_fact_rets: DataFrame of factor returns with shape (n_timestamps, n_factors).
        half_life: Half-life in time units (e.g., days). Must be positive.
            Recommendations for half-life based on the horizon:
            | Horizon (days) | Recommended half-life (days) |
            |----------------|------------------------------|
            | 30             | 10                           |
            | 60             | 20                           |
            | 90             | 30                           |
            | 180            | 60                           |
            | 365            | 120                          |
        lambda_: Decay factor (e.g., 0.985). Must be between 0 and 1.
        min_timestamps: Minimum number of timestamps required for regression. Defaults to 10.

    Returns:
        DataFrame of shape (n_factors + 1, n_assets) containing the computed betas.

    Raises:
        ValueError: If inputs are empty, have insufficient data, mismatched rows,
            excessive NaNs, or near-zero variance in df_fact_rets.
    """
    # Check for empty inputs
    if df_asset_rets.empty:
        logger.error("Input DataFrame df_asset_rets is empty.")
        raise ValueError("Input DataFrame df_asset_rets is empty.")
    if df_fact_rets.empty:
        logger.error("Input DataFrame df_fact_rets is empty.")
        raise ValueError("Input DataFrame df_fact_rets is empty.")

    # Check for mismatched row counts
    if df_asset_rets.shape[0] != df_fact_rets.shape[0]:
        logger.error(
            "Mismatched row counts: df_asset_rets has %d rows, df_fact_rets has %d rows.",
            df_asset_rets.shape[0],
            df_fact_rets.shape[0],
        )
        raise ValueError(
            "Mismatched row counts: "
            f"df_asset_rets has {df_asset_rets.shape[0]} rows, "
            f"df_fact_rets has {df_fact_rets.shape[0]} rows."
        )

    # Align df_asset_rets and df_fact_rets indices and handle NaNs
    df_combined: pd.DataFrame = pd.concat(
        [df_asset_rets, df_fact_rets], axis=1, join="inner"
    )
    if df_combined.empty:
        logger.error(
            "No overlapping timestamps between df_asset_rets and df_fact_rets."
        )
        raise ValueError(
            "No overlapping timestamps between df_asset_rets and df_fact_rets."
        )

    df_y_clean: pd.DataFrame = df_combined[df_asset_rets.columns].dropna()
    df_x_clean: pd.DataFrame = df_combined[df_fact_rets.columns].loc[df_y_clean.index]

    n_timestamps, _ = df_y_clean.shape

    # Log data cleaning results
    if len(df_asset_rets) != n_timestamps:
        dropped_rows = len(df_asset_rets) - n_timestamps
        logger.warning(
            "Dropped %d rows due to NaNs or index misalignment. "
            "Remaining timestamps: %d",
            dropped_rows,
            n_timestamps,
        )
        # Check for excessive NaN dropping
        if dropped_rows > 0.5 * len(df_asset_rets):
            logger.error(
                "Excessive data loss: %d rows dropped of %d.",
                dropped_rows,
                len(df_asset_rets),
            )
            raise ValueError(
                f"Excessive data loss: {dropped_rows} rows dropped of {len(df_asset_rets)}."
            )

    # Check minimum timestamps
    if n_timestamps < min_timestamps:
        logger.error(
            "Insufficient data: %d timestamps available, minimum required is %d.",
            n_timestamps,
            min_timestamps,
        )
        raise ValueError(
            f"Insufficient data: {n_timestamps} timestamps available, "
            f"minimum required is {min_timestamps}."
        )

    # Check for near-zero variance in df_fact_rets
    if df_x_clean.var().min() < 1e-10:
        logger.error("One or more factors in df_fact_rets have near-zero variance.")
        raise ValueError("One or more factors in df_fact_rets have near-zero variance.")

    # Calculate weights
    weights: np.ndarray = exponential_weights(
        n_timestamps, half_life=half_life, lambda_=lambda_
    )
    sqrt_weights: np.ndarray = np.sqrt(weights)

    beta_matrix: pd.DataFrame = pd.DataFrame(
        index=["Intercept"] + list(df_x_clean.columns), columns=df_y_clean.columns
    )

    x_weighted: pd.DataFrame = df_x_clean.multiply(sqrt_weights, axis=0)
    for asset in df_y_clean.columns:
        y: np.ndarray = df_y_clean[asset].values
        y_weighted: np.ndarray = y * sqrt_weights

        x_w_const: pd.DataFrame = sm.add_constant(x_weighted)
        rlm_model = sm.RLM(y_weighted, x_w_const, M=sm.robust.norms.HuberT())
        rlm_results = rlm_model.fit()

        beta_matrix[asset] = rlm_results.params

    return beta_matrix
