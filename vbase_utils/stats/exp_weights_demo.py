import matplotlib.pyplot as plt
import numpy as np
from vbase_utils.stats.robust_betas import exponential_weights

def test_plot_exponential_weights():
    n = 100  # number of lookback periods
    half_lives = [10, 20, 50]   # different half-life settings
    lambdas = [0.95, 0.98]      # explicit lambda values

    plt.figure(figsize=(10, 6))

    # Plot half-life versions
    for hl in half_lives:
        w = exponential_weights(n=n, half_life=hl)
        plt.plot(range(n), w, label=f"Half-life = {hl}")

    # Plot lambda versions
    for lam in lambdas:
        w = exponential_weights(n=n, lambda_=lam)
        plt.plot(range(n), w, linestyle="--", label=f"λ = {lam}")

    plt.title("Exponential Decay Weights (vbase_utils)")
    plt.xlabel("Time steps (0 = oldest, n = most recent)")
    plt.ylabel("Weight")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"./exp_weights_demo.png")
    plt.close()


if __name__ == "__main__":
    test_plot_exponential_weights()
