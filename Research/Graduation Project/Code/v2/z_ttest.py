import math
from scipy.stats import t

def correlation_significance(r, n):
    # Check input validity
    if not -1 <= r <= 1:
        raise ValueError("Correlation coefficient r must be between -1 and 1.")
    if n <= 2:
        raise ValueError("Number of data points n must be greater than 2.")
    
    # Compute t-statistic
    t_stat = r * math.sqrt((n - 2) / (1 - r**2))
    
    # Degrees of freedom
    df = n - 2
    
    # Two-tailed p-value
    p_value = 2 * (1 - t.cdf(abs(t_stat), df))
    
    return t_stat, p_value

# Example usage:
if __name__ == "__main__":
    r = 0.1625   # example correlation coefficient
    n = 86    # example number of data points

    t_stat, p_value = correlation_significance(r, n)
    print(f"r = {r}, n = {n}")
    print(f"t-statistic = {t_stat:.4f}")
    print(f"p-value = {p_value:.6f}")

    if p_value < 0.05:
        print("The correlation is statistically significant (p < 0.05).")
    else:
        print("The correlation is not statistically significant (p >= 0.05).")
