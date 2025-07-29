import numpy as np

class HeterogeneousNetworkGenerator:
    """
    A class to generate series of numbers following specified statistical distributions.
    This is a Python implementation inspired by the MATLAB Fun_Hete_g class.
    """
    @staticmethod
    def generate_series(series_length: int, distribution_type: str, mean_value: float, std_dev: float, truncation_limits: tuple[float, float]) -> np.ndarray:
        """
        Generates a random series following different distributions.

        Args:
            series_length: Length of the output series.
            distribution_type: The type of distribution for the series.
                               Supported: 'lognormal', 'lognormal_int'.
            mean_value: Mean value of the distribution.
            std_dev: Standard deviation of the distribution.
            truncation_limits: A tuple (min_val, max_val) for truncation.

        Returns:
            A NumPy array with the generated series.

        Raises:
            ValueError: If an unsupported distribution_type is provided.
        """
        if distribution_type not in ['lognormal', 'lognormal_int']:
            raise ValueError(f"Unsupported distribution_type: {distribution_type}")

        # Parameters for lognormal distribution from mean and std dev
        # of the non-logarithmized variable.
        mean_sq = mean_value**2
        variance = std_dev**2
        mu = np.log(mean_sq / np.sqrt(variance + mean_sq))
        sigma = np.sqrt(np.log(variance / mean_sq + 1))

        final_series = np.array([])
        min_val, max_val = truncation_limits

        # Generate numbers in a loop to ensure we get enough values after truncation
        while len(final_series) < series_length:
            # Generate a batch of random numbers, size is dynamically adjusted
            needed = series_length - len(final_series)
            # Generate more than needed to account for truncation
            batch_size = max(100, needed * 5)

            random_series = np.random.lognormal(mu, sigma, batch_size)

            if distribution_type == 'lognormal_int':
                random_series = np.round(random_series)

            # Truncate the random numbers to be within the specified range
            truncated_series = random_series[(random_series > min_val) & (random_series < max_val)]

            # Append to the final series
            final_series = np.concatenate((final_series, truncated_series))

        # Return the exact number of elements requested
        return final_series[:series_length]