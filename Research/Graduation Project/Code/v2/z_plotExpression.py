import numpy as np
import matplotlib.pyplot as plt

def safe_atan2(y, x):
    """Two-argument arctangent"""
    return np.arctan2(y, x)

def evolved_equation(count: float, complexity: float, p0: float) -> float:
    """
    Computes:
    wavg(count, p0) ^ cosh(sigmoid(wavg(complexity, sqrt(cos((complexity + complexity) - count)))))
    """
    # Step 1: Inner cosine term
    inner_cos = np.cos((complexity + complexity) - count)
    
    # Step 2: Square root (safe)
    sqrt_term = np.sqrt(np.abs(inner_cos))
    
    # Step 3: Weighted average with complexity
    wavg_inner = 0.7 * complexity + 0.3 * sqrt_term
    
    # Step 4: Sigmoid
    sigmoid_inner = 1.0 / (1.0 + np.exp(-np.clip(wavg_inner, -100, 100)))
    
    # Step 5: Hyperbolic cosine
    cosh_inner = np.cosh(np.clip(sigmoid_inner, -100, 100))
    
    # Step 6: Weighted average of count and p0
    wavg_outer = 0.7 * count + 0.3 * p0
    
    # Step 7: Final power
    result = np.power(wavg_outer, cosh_inner)
    
    return result

p1 = 1
count = np.logspace(0, 11, 1000)

complexities = np.arange(0.0000001, 1.01, 0.01)

for i, complexity in enumerate(complexities):
    y = evolved_equation(count, complexity, p1)
    label = f'complexity={complexity:.7f}' if i == 0 or i == len(complexities) - 1 else None
    plt.plot(count, y, color=plt.cm.viridis(i / len(complexities)), label=label)

plt.title(r'$(p1 + count)^{atan2(count, complexity)}$ for various complexities')
plt.xlabel('count')
plt.ylabel('result')
plt.xscale('log')
plt.grid(True)
plt.legend()
plt.show()
