import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from esn_gen_hete import HeterogeneousNetworkGenerator

def test_lognormal_distribution():
    """
    Tests the lognormal distribution generator by creating and displaying
    a histogram and a Q-Q plot.
    """
    # Parameters for the data generation (user's values)
    series_len = 5000
    dist_type = 'lognormal'
    mean = 20.0
    std = 5.0
    # Adjust limits to be appropriate for the mean and std dev
    # A reasonable range would be e.g., mean +/- 4*std, ensuring positivity.
    limits = (1, 100 + mean + 4 * std)

    # Generate the data using the class
    try:
        generated_series = HeterogeneousNetworkGenerator.generate_series(
            series_len, dist_type, mean, std, limits
        )
    except ValueError as e:
        print(f"Error generating series: {e}")
        return

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Histogram
    ax1.hist(generated_series, bins=50, density=True, alpha=0.7, label='Generated Data')
    
    # Overlay the theoretical PDF for comparison
    shape, loc, scale = stats.lognorm.fit(generated_series, floc=0)
    x = np.linspace(generated_series.min(), generated_series.max(), 200)
    pdf = stats.lognorm.pdf(x, shape, loc, scale)
    ax1.plot(x, pdf, 'r-', lw=2, label='Theoretical PDF')
    ax1.set_title('Histogram of Generated Data')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Density')
    ax1.legend()

    # 2. Q-Q Plot
    stats.probplot(generated_series, dist=stats.lognorm, sparams=(shape,), plot=ax2)
    ax2.set_title('Q-Q Plot vs. Log-Normal Distribution')

    plt.suptitle('Log-Normal Distribution Verification', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()

if __name__ == '__main__':
    test_lognormal_distribution()
