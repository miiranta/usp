"""
Genetic Algorithm Building Blocks for Equation Evolution

This module provides a comprehensive set of mathematical operations
that can be used to build and evolve equations for benchmark prediction.
"""

import numpy as np
from typing import Callable, Dict, List, Tuple, Any
import random

# ============================================================================
# BASIC BUILDING BLOCKS
# ============================================================================

class BuildingBlock:
    """Represents a single mathematical operation/function"""
    def __init__(self, name: str, func: Callable, arity: int, 
                 complexity_cost: float = 1.0, symbol: str = None):
        self.name = name
        self.func = func
        self.arity = arity  # Number of arguments
        self.complexity_cost = complexity_cost
        self.symbol = symbol or name
    
    def __call__(self, *args):
        return self.func(*args)

# ============================================================================
# NULLARY OPERATIONS (0 arguments - terminals/variables)
# ============================================================================

NULLARY_OPS = {
    'complexity': BuildingBlock('complexity', lambda: 'complexity', 0, 0.0, 'C'),
    'count': BuildingBlock('count', lambda: 'count', 0, 0.0, 'N'),
}

# ============================================================================
# UNARY OPERATIONS (1 argument)
# ============================================================================

def safe_sqrt(x):
    """Safe square root that handles negative values"""
    return np.sqrt(np.abs(x))

def safe_log(x):
    """Safe logarithm that handles non-positive values"""
    return np.log(np.abs(x) + 1)

def safe_log2(x):
    """Safe log base 2"""
    return np.log2(np.abs(x) + 1)

def safe_log10(x):
    """Safe log base 10"""
    return np.log10(np.abs(x) + 1)

def safe_exp(x):
    """Safe exponential that clips extreme values"""
    return np.exp(np.clip(x, -100, 100))

def safe_sin(x):
    """Safe sine function"""
    return np.sin(x)

def safe_cos(x):
    """Safe cosine function"""
    return np.cos(x)

def safe_tan(x):
    """Safe tangent with clipping"""
    return np.tan(np.clip(x, -np.pi/2 + 0.1, np.pi/2 - 0.1))

def safe_reciprocal(x):
    """Safe reciprocal: 1/x"""
    return 1.0 / (np.abs(x) + 1e-6)

def safe_square(x):
    """Square function"""
    return x ** 2

def safe_cube(x):
    """Cube function"""
    return x ** 3

def safe_abs(x):
    """Absolute value"""
    return np.abs(x)

def safe_neg(x):
    """Negation"""
    return -x

def safe_sigmoid(x):
    """Sigmoid function: 1/(1+e^-x)"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -100, 100)))

def safe_tanh(x):
    """Hyperbolic tangent"""
    return np.tanh(x)

def safe_cbrt(x):
    """Cube root (handles negative values)"""
    return np.sign(x) * np.abs(x) ** (1/3)

def safe_fourth_root(x):
    """Fourth root"""
    return np.abs(x) ** 0.25

def safe_floor(x):
    """Floor function"""
    return np.floor(x)

def safe_ceil(x):
    """Ceiling function"""
    return np.ceil(x)

def safe_round(x):
    """Round function"""
    return np.round(x)

def safe_arcsin(x):
    """Arcsine (clipped to valid range)"""
    return np.arcsin(np.clip(x, -1, 1))

def safe_arccos(x):
    """Arccosine (clipped to valid range)"""
    return np.arccos(np.clip(x, -1, 1))

def safe_arctan(x):
    """Arctangent"""
    return np.arctan(x)

def safe_sinh(x):
    """Hyperbolic sine"""
    return np.sinh(np.clip(x, -100, 100))

def safe_cosh(x):
    """Hyperbolic cosine"""
    return np.cosh(np.clip(x, -100, 100))

def safe_fifth_root(x):
    """Fifth root (handles negative values)"""
    return np.sign(x) * np.abs(x) ** 0.2

def safe_sixth_root(x):
    """Sixth root"""
    return np.abs(x) ** (1/6)

UNARY_OPS = {
    'sqrt': BuildingBlock('sqrt', safe_sqrt, 1, 1.0, '√'),
    'log': BuildingBlock('log', safe_log, 1, 1.2, 'ln'),
    'log2': BuildingBlock('log2', safe_log2, 1, 1.2, 'log₂'),
    'log10': BuildingBlock('log10', safe_log10, 1, 1.2, 'log₁₀'),
    'exp': BuildingBlock('exp', safe_exp, 1, 1.5, 'exp'),
    'sin': BuildingBlock('sin', safe_sin, 1, 1.3, 'sin'),
    'cos': BuildingBlock('cos', safe_cos, 1, 1.3, 'cos'),
    'tan': BuildingBlock('tan', safe_tan, 1, 1.4, 'tan'),
    'arcsin': BuildingBlock('arcsin', safe_arcsin, 1, 1.3, 'asin'),
    'arccos': BuildingBlock('arccos', safe_arccos, 1, 1.3, 'acos'),
    'arctan': BuildingBlock('arctan', safe_arctan, 1, 1.3, 'atan'),
    'sinh': BuildingBlock('sinh', safe_sinh, 1, 1.4, 'sinh'),
    'cosh': BuildingBlock('cosh', safe_cosh, 1, 1.4, 'cosh'),
    'reciprocal': BuildingBlock('reciprocal', safe_reciprocal, 1, 1.1, '1/'),
    'square': BuildingBlock('square', safe_square, 1, 0.8, '²'),
    'cube': BuildingBlock('cube', safe_cube, 1, 0.9, '³'),
    'abs': BuildingBlock('abs', safe_abs, 1, 0.7, '|·|'),
    'neg': BuildingBlock('neg', safe_neg, 1, 0.5, '-'),
    'sigmoid': BuildingBlock('sigmoid', safe_sigmoid, 1, 1.6, 'σ'),
    'tanh': BuildingBlock('tanh', safe_tanh, 1, 1.4, 'tanh'),
    'cbrt': BuildingBlock('cbrt', safe_cbrt, 1, 1.1, '∛'),
    'fourth_root': BuildingBlock('fourth_root', safe_fourth_root, 1, 1.2, '∜'),
    'fifth_root': BuildingBlock('fifth_root', safe_fifth_root, 1, 1.2, '∜'),
    'sixth_root': BuildingBlock('sixth_root', safe_sixth_root, 1, 1.2, '∜'),
    'floor': BuildingBlock('floor', safe_floor, 1, 0.8, '⌊·⌋'),
    'ceil': BuildingBlock('ceil', safe_ceil, 1, 0.8, '⌈·⌉'),
    'round': BuildingBlock('round', safe_round, 1, 0.8, 'round'),
}

# ============================================================================
# BINARY OPERATIONS (2 arguments)
# ============================================================================

def safe_add(x, y):
    """Addition"""
    return x + y

def safe_sub(x, y):
    """Subtraction"""
    return x - y

def safe_mul(x, y):
    """Multiplication"""
    return x * y

def safe_div(x, y):
    """Safe division"""
    return x / (np.abs(y) + 1e-6)

def safe_pow(x, y):
    """Safe power function"""
    return np.abs(x) ** np.clip(y, -10, 10)

def safe_mod(x, y):
    """Safe modulo"""
    return np.mod(x, np.abs(y) + 1e-6)

def safe_min(x, y):
    """Minimum of two values"""
    return np.minimum(x, y)

def safe_max(x, y):
    """Maximum of two values"""
    return np.maximum(x, y)

def safe_avg(x, y):
    """Average of two values"""
    return (x + y) / 2.0

def safe_weighted_avg(x, y):
    """Weighted average (0.7*x + 0.3*y)"""
    return 0.7 * x + 0.3 * y

def safe_geometric_mean(x, y):
    """Geometric mean"""
    return np.sqrt(np.abs(x * y))

def safe_harmonic_mean(x, y):
    """Harmonic mean"""
    return 2.0 / (1.0/(np.abs(x) + 1e-6) + 1.0/(np.abs(y) + 1e-6))

def safe_hypot(x, y):
    """Euclidean distance: sqrt(x^2 + y^2)"""
    return np.sqrt(x**2 + y**2)

def safe_atan2(y, x):
    """Two-argument arctangent"""
    return np.arctan2(y, x)

def safe_gcd_like(x, y):
    """GCD-like operation for floats (using remainder)"""
    x_abs = np.abs(x) + 0.1
    y_abs = np.abs(y) + 0.1
    return np.minimum(x_abs, y_abs)

def safe_lcm_like(x, y):
    """LCM-like operation for floats"""
    x_abs = np.abs(x) + 0.1
    y_abs = np.abs(y) + 0.1
    return np.maximum(x_abs, y_abs)

BINARY_OPS = {
    'add': BuildingBlock('add', safe_add, 2, 0.5, '+'),
    'sub': BuildingBlock('sub', safe_sub, 2, 0.5, '-'),
    'mul': BuildingBlock('mul', safe_mul, 2, 0.6, '×'),
    'div': BuildingBlock('div', safe_div, 2, 0.8, '/'),
    'pow': BuildingBlock('pow', safe_pow, 2, 1.0, '^'),
    'mod': BuildingBlock('mod', safe_mod, 2, 1.2, 'mod'),
    'min': BuildingBlock('min', safe_min, 2, 0.9, 'min'),
    'max': BuildingBlock('max', safe_max, 2, 0.9, 'max'),
    'avg': BuildingBlock('avg', safe_avg, 2, 0.7, 'avg'),
    'weighted_avg': BuildingBlock('weighted_avg', safe_weighted_avg, 2, 0.8, 'wavg'),
    'geometric_mean': BuildingBlock('geometric_mean', safe_geometric_mean, 2, 1.1, 'gmean'),
    'harmonic_mean': BuildingBlock('harmonic_mean', safe_harmonic_mean, 2, 1.2, 'hmean'),
    'hypot': BuildingBlock('hypot', safe_hypot, 2, 1.0, 'hypot'),
    'atan2': BuildingBlock('atan2', safe_atan2, 2, 1.3, 'atan2'),
    'gcd_like': BuildingBlock('gcd_like', safe_gcd_like, 2, 1.1, 'gcd'),
    'lcm_like': BuildingBlock('lcm_like', safe_lcm_like, 2, 1.1, 'lcm'),
}

# ============================================================================
# TERNARY OPERATIONS (3 arguments)
# ============================================================================

def safe_if_then_else(condition, if_true, if_false):
    """Conditional operation"""
    return np.where(condition > 0, if_true, if_false)

def safe_clamp(x, min_val, max_val):
    """Clamp x between min_val and max_val"""
    return np.clip(x, min_val, max_val)

def safe_lerp(a, b, t):
    """Linear interpolation: a + t*(b-a)"""
    t_clipped = np.clip(t, 0, 1)
    return a + t_clipped * (b - a)

def safe_weighted_sum(a, b, c):
    """Weighted sum: 0.5*a + 0.3*b + 0.2*c"""
    return 0.5 * a + 0.3 * b + 0.2 * c

TERNARY_OPS = {
    'if_then_else': BuildingBlock('if_then_else', safe_if_then_else, 3, 1.5, 'if'),
    'clamp': BuildingBlock('clamp', safe_clamp, 3, 1.1, 'clamp'),
    'lerp': BuildingBlock('lerp', safe_lerp, 3, 1.2, 'lerp'),
    'weighted_sum': BuildingBlock('weighted_sum', safe_weighted_sum, 3, 0.9, 'wsum'),
}

# ============================================================================
# ALL OPERATIONS COMBINED
# ============================================================================

ALL_OPS = {
    **NULLARY_OPS,
    **UNARY_OPS,
    **BINARY_OPS,
    **TERNARY_OPS
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_operations_by_arity(arity: int) -> Dict[str, BuildingBlock]:
    """Get all operations with a specific arity"""
    return {k: v for k, v in ALL_OPS.items() if v.arity == arity}

def get_random_operation(arity: int = None) -> BuildingBlock:
    """Get a random operation, optionally filtered by arity"""
    if arity is not None:
        ops = get_operations_by_arity(arity)
    else:
        ops = ALL_OPS
    
    if not ops:
        raise ValueError(f"No operations with arity {arity}")
    
    return random.choice(list(ops.values()))

def get_terminal() -> str:
    """Get a random terminal (variable)"""
    return random.choice(['complexity', 'count'])

# ============================================================================
# CONSTANT GENERATION
# ============================================================================

def generate_random_constant() -> float:
    """Generate a random constant for equation parameters"""
    # Mix of different ranges to cover various scales
    choice = random.random()
    
    if choice < 0.3:
        # Small constants around 1
        return random.uniform(0.1, 3.0)
    elif choice < 0.5:
        # Larger constants
        return random.uniform(1.0, 10.0)
    elif choice < 0.7:
        # Very small constants
        return random.uniform(0.001, 0.1)
    elif choice < 0.85:
        # Negative constants
        return random.uniform(-5.0, -0.1)
    else:
        # Zero or near-zero
        return random.uniform(-0.5, 0.5)

# ============================================================================
# STATISTICS
# ============================================================================

def print_building_blocks_summary():
    """Print a summary of available building blocks"""
    print("\n" + "="*70)
    print("BUILDING BLOCKS SUMMARY")
    print("="*70)
    
    print(f"\nNullary Operations (Variables): {len(NULLARY_OPS)}")
    for name, op in NULLARY_OPS.items():
        print(f"  • {name} ({op.symbol})")
    
    print(f"\nUnary Operations: {len(UNARY_OPS)}")
    for name, op in UNARY_OPS.items():
        print(f"  • {name} ({op.symbol}) - complexity: {op.complexity_cost:.1f}")
    
    print(f"\nBinary Operations: {len(BINARY_OPS)}")
    for name, op in BINARY_OPS.items():
        print(f"  • {name} ({op.symbol}) - complexity: {op.complexity_cost:.1f}")
    
    print(f"\nTernary Operations: {len(TERNARY_OPS)}")
    for name, op in TERNARY_OPS.items():
        print(f"  • {name} ({op.symbol}) - complexity: {op.complexity_cost:.1f}")
    
    print(f"\nTotal Operations: {len(ALL_OPS)}")
    print("="*70 + "\n")

if __name__ == '__main__':
    print_building_blocks_summary()
