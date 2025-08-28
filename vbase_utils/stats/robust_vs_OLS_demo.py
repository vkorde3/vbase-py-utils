import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from vbase_utils.stats.robust_betas import robust_betas

# Simulate asset returns with beta ~1.5
np.random.seed(42)
n_obs = 200
market_rets = np.random.normal(0, 0.02, n_obs)
asset_rets = 1.5 * market_rets + np.random.normal(0, 0.01, n_obs)

# Inject one outlier
asset_rets[50] += 0.2

df_fact_rets = pd.DataFrame({"market": market_rets})
df_asset_rets = pd.DataFrame({"asset": asset_rets})

# Robust regression
robust_res = robust_betas(
    df_asset_rets=df_asset_rets,
    df_fact_rets=df_fact_rets,
    lambda_=0.94,
    min_timestamps=10,
)

# OLS regression
X = sm.add_constant(df_fact_rets["market"])
ols_model = sm.OLS(df_asset_rets["asset"], X).fit()

# Extract betas
beta_robust = robust_res.loc["market", "asset"]
beta_ols = ols_model.params["market"]
alpha_ols = ols_model.params["const"]

print(f"Robust Beta: {beta_robust:.4f}")
print(f"OLS Beta: {beta_ols:.4f}")

# Create fit lines
x_line = np.linspace(market_rets.min(), market_rets.max(), 100)

# OLS fit line (with intercept)
y_ols = alpha_ols + beta_ols * x_line

# Robust fit line (assume intercept ≈ 0)
y_robust = beta_robust * x_line

# --- Plot ---
plt.figure(figsize=(8,6))

# Plot all synthetic data points as blue crosses
plt.scatter(market_rets, asset_rets, marker="x", color="blue", alpha=0.6, label="Data")

# Highlight outlier in red cross
plt.scatter(market_rets[50], asset_rets[50], marker="x", color="red", s=100, label="Outlier")

# Plot OLS fit (blue line)
plt.plot(x_line, y_ols, color="blue", linewidth=2, label=f"OLS Fit (β={beta_ols:.2f})")

# Plot Robust fit (red line)
plt.plot(x_line, y_robust, color="red", linewidth=2, linestyle="--", label=f"Robust Fit (β={beta_robust:.2f})")

plt.xlabel("Market Returns")
plt.ylabel("Asset Returns")
plt.title("OLS vs Robust Regression with Outlier")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig("robust_vs_ols_fits.png", dpi=300)
plt.show()
