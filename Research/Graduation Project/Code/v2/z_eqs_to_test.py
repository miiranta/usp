import numpy as np

EQUATIONS_TO_TEST = {
    # ========== SINGLE VARIABLE - COUNT ONLY (EXTENDED) ==========
    'count_only_1': {
        'func': lambda complexity, count, a, b, c: a * (count ** b) + c,
        'name': 'a·count^b + c',
        'initial_guess': [1, 1.5, 0]
    },
    'count_only_2': {
        'func': lambda complexity, count, a, b, c: a * (count ** b) + c * np.sqrt(count),
        'name': 'a·count^b + c·√count',
        'initial_guess': [1, 1.5, 1]
    },
    'count_only_3': {
        'func': lambda complexity, count, a, b, c, d: a * (count ** b) + c * count + d,
        'name': 'a·count^b + c·count + d',
        'initial_guess': [1, 1.5, 1, 0]
    },
    'count_only_4': {
        'func': lambda complexity, count, a, b, c, d: a * (count ** b) + c * np.sqrt(count) + d * np.log(count + 1),
        'name': 'a·count^b + c·√count + d·ln(count+1)',
        'initial_guess': [1, 1.5, 1, 1]
    },
    'count_only_5': {
        'func': lambda complexity, count, a, b: a * (count ** b),
        'name': 'a·count^b',
        'initial_guess': [1, 1.5]
    },
    'count_only_6': {
        'func': lambda complexity, count, a, b, c: a * np.log(count + 1) + b * np.sqrt(count) + c,
        'name': 'a·ln(count+1) + b·√count + c',
        'initial_guess': [1, 1, 0]
    },
    'count_only_7': {
        'func': lambda complexity, count, a, b: a * count + b,
        'name': 'a·count + b',
        'initial_guess': [1, 0]
    },
    
    # ========== SINGLE VARIABLE - COMPLEXITY ONLY (EXTENDED) ==========
    'complexity_only_1': {
        'func': lambda complexity, count, a, b, c: a * (complexity ** b) + c,
        'name': 'a·complexity^b + c',
        'initial_guess': [1, 0.5, 0]
    },
    'complexity_only_2': {
        'func': lambda complexity, count, a, b, c, d: a * (complexity ** b) + c * np.sqrt(complexity) + d,
        'name': 'a·complexity^b + c·√complexity + d',
        'initial_guess': [1, 0.5, 1, 0]
    },
    'complexity_only_3': {
        'func': lambda complexity, count, a, b: a * complexity + b,
        'name': 'a·complexity + b',
        'initial_guess': [1, 0]
    },
    'complexity_only_4': {
        'func': lambda complexity, count, a, b, c: a * np.sqrt(complexity) + b * np.log(complexity + 1) + c,
        'name': 'a·√complexity + b·ln(complexity+1) + c',
        'initial_guess': [1, 1, 0]
    },
    
    # ========== ULTRA-SIMPLE CORE (2-4 parameters) ==========
    'ultra_simple_1': {
        'func': lambda complexity, count, a, b, c: a * (complexity ** b) * (count ** c),
        'name': 'a·complexity^b·count^c',
        'initial_guess': [1, 0.3, 1.4]
    },
    'ultra_simple_2': {
        'func': lambda complexity, count, a, b, c, d: a * (complexity ** b) * (count ** c) + d,
        'name': 'a·complexity^b·count^c + d',
        'initial_guess': [1, 0.3, 1.4, 0]
    },
    'ultra_simple_3': {
        'func': lambda complexity, count, a, b, c, d: a * (complexity ** b) * (count ** c) + d * count,
        'name': 'a·complexity^b·count^c + d·count',
        'initial_guess': [1, 0.3, 1.4, 0.5]
    },
    'ultra_simple_4': {
        'func': lambda complexity, count, a, b, c, d: a * (complexity ** b) * (count ** c) + d * complexity,
        'name': 'a·complexity^b·count^c + d·complexity',
        'initial_guess': [1, 0.3, 1.4, 1]
    },
    
    # ========== SIMPLIFIED CHAMPIONS (4-5 parameters) ==========
    'simple_champion_1': {
        'func': lambda complexity, count, a, b, c, d, e: a * (complexity ** b) * (count ** c) + d * np.sqrt(count) + e,
        'name': 'a·complexity^b·count^c + d·√count + e',
        'initial_guess': [1, 0.3, 1.4, 1, 0]
    },
    'simple_champion_2': {
        'func': lambda complexity, count, a, b, c, d, e: a * (complexity ** b) * (count ** c) + d * complexity + e,
        'name': 'a·complexity^b·count^c + d·complexity + e',
        'initial_guess': [1, 0.3, 1.4, 1, 0]
    },
    'simple_champion_3': {
        'func': lambda complexity, count, a, b, c, d, e: a * (complexity ** b) * (count ** c) + d * np.log(count + 1) + e,
        'name': 'a·complexity^b·count^c + d·ln(count+1) + e',
        'initial_guess': [1, 0.3, 1.4, 1, 0]
    },
    'simple_champion_4': {
        'func': lambda complexity, count, a, b, c, d: a * (complexity ** b) * (count ** c) + d * count,
        'name': 'a·complexity^b·count^c + d·count',
        'initial_guess': [1, 0.3, 1.4, 0.5]
    },
    
    # ========== BEST PREVIOUS (6 parameters - THE STANDARD) ==========
    'standard_sqrt_count': {
        'func': lambda complexity, count, a, b, c, d, e, f: a * (complexity ** b) * (count ** c) + d * count + e * np.sqrt(count) + f,
        'name': 'a·complexity^b·count^c + d·count + e·√count + f',
        'initial_guess': [1, 0.3, 1.4, 0.5, 1, 0]
    },
    'standard_complexity': {
        'func': lambda complexity, count, a, b, c, d, e, f: a * (complexity ** b) * (count ** c) + d * count + e * complexity + f,
        'name': 'a·complexity^b·count^c + d·count + e·complexity + f',
        'initial_guess': [1, 0.3, 1.4, 0.5, 1, 0]
    },
    
    # ========== SIMPLIFIED BEST PERFORMER (5-6 params, no offset) ==========
    'simplified_best_1': {
        'func': lambda complexity, count, a, b, c, d, e: a * (complexity ** b) * (count ** c) + d * count + e * complexity,
        'name': 'a·complexity^b·count^c + d·count + e·complexity (no offset)',
        'initial_guess': [1, 0.3, 1.4, 0.5, 1]
    },
    'simplified_best_2': {
        'func': lambda complexity, count, a, b, c, d, e: a * (complexity ** b) * (count ** c) + d * np.sqrt(count) + e * complexity,
        'name': 'a·complexity^b·count^c + d·√count + e·complexity (no offset)',
        'initial_guess': [1, 0.3, 1.4, 1, 1]
    },
    'simplified_best_3': {
        'func': lambda complexity, count, a, b, c, d, e: a * (complexity ** b) * (count ** c) + d * count + e * np.sqrt(count),
        'name': 'a·complexity^b·count^c + d·count + e·√count (no offset)',
        'initial_guess': [1, 0.3, 1.4, 0.5, 1]
    },
    
    # ========== SPECIFIC POWER VALUES (testing optimal exponents) ==========
    'fixed_power_1': {
        'func': lambda complexity, count, a, b, c: a * (complexity ** 0.2) * (count ** 1.4) + b * count + c,
        'name': 'a·complexity^0.2·count^1.4 + b·count + c',
        'initial_guess': [1, 0.5, 0]
    },
    'fixed_power_2': {
        'func': lambda complexity, count, a, b, c: a * (complexity ** 0.15) * (count ** 1.5) + b * np.sqrt(count) + c,
        'name': 'a·complexity^0.15·count^1.5 + b·√count + c',
        'initial_guess': [1, 1, 0]
    },
    'fixed_power_3': {
        'func': lambda complexity, count, a, b, c, d: a * (complexity ** 0.25) * (count ** 1.3) + b * count + c * np.sqrt(count) + d,
        'name': 'a·complexity^0.25·count^1.3 + b·count + c·√count + d',
        'initial_guess': [1, 0.5, 1, 0]
    },
    'fixed_power_4': {
        'func': lambda complexity, count, a, b, c: a * (complexity ** 0.1) * (count ** 1.6) + b * count + c,
        'name': 'a·complexity^0.1·count^1.6 + b·count + c',
        'initial_guess': [1, 0.5, 0]
    },
    'fixed_power_5': {
        'func': lambda complexity, count, a, b, c: a * (complexity ** 0.3) * (count ** 1.4) + b * np.sqrt(count) + c,
        'name': 'a·complexity^0.3·count^1.4 + b·√count + c',
        'initial_guess': [1, 1, 0]
    },
    'fixed_power_6': {
        'func': lambda complexity, count, a, b, c: a * (complexity ** 0.35) * (count ** 1.35) + b * count + c,
        'name': 'a·complexity^0.35·count^1.35 + b·count + c',
        'initial_guess': [1, 0.5, 0]
    },
    'fixed_power_7': {
        'func': lambda complexity, count, a, b, c: a * (complexity ** 0.4) * (count ** 1.2) + b * np.sqrt(count) + c,
        'name': 'a·complexity^0.4·count^1.2 + b·√count + c',
        'initial_guess': [1, 1, 0]
    },
    'fixed_power_8': {
        'func': lambda complexity, count, a, b, c: a * (complexity ** 0.5) * (count ** 1.0) + b * count + c,
        'name': 'a·complexity^0.5·count^1.0 + b·count + c',
        'initial_guess': [1, 0.5, 0]
    },
    
    # ========== OPTIMIZED 7-PARAMETER VERSIONS (trying to hit 0.6) ==========
    'optimal_7param_1': {
        'func': lambda complexity, count, a, b, c, d, e, f, g: a * (complexity ** b) * (count ** c) + d * count + e * complexity + f * np.sqrt(complexity) + g,
        'name': 'a·complexity^b·count^c + d·count + e·complexity + f·√complexity + g',
        'initial_guess': [1, 0.3, 1.4, 0.5, 1, 0.5, 0]
    },
    'optimal_7param_2': {
        'func': lambda complexity, count, a, b, c, d, e, f, g: a * (complexity ** b) * (count ** c) + d * count + e * np.sqrt(count) + f * complexity + g,
        'name': 'a·complexity^b·count^c + d·count + e·√count + f·complexity + g',
        'initial_guess': [1, 0.3, 1.4, 0.5, 1, 1, 0]
    },
    'optimal_7param_3': {
        'func': lambda complexity, count, a, b, c, d, e, f, g: a * (complexity ** b) * (count ** c) + d * count + e * np.sqrt(count) + f * np.log(complexity + 1) + g,
        'name': 'a·complexity^b·count^c + d·count + e·√count + f·ln(complexity+1) + g',
        'initial_guess': [1, 0.3, 1.4, 0.5, 1, 0.5, 0]
    },
    'optimal_7param_4': {
        'func': lambda complexity, count, a, b, c, d, e, f, g: a * (complexity ** b) * (count ** c) + d * count + e * np.sqrt(count) + f * np.sqrt(complexity) + g,
        'name': 'a·complexity^b·count^c + d·count + e·√count + f·√complexity + g',
        'initial_guess': [1, 0.3, 1.4, 0.5, 1, 0.5, 0]
    },
    
    # ========== DOUBLE ENHANCEMENT (mixing best features) ==========
    'double_enhance_1': {
        'func': lambda complexity, count, a, b, c, d, e, f: a * (complexity ** b) * (count ** c) + d * np.sqrt(count) + e * complexity + f,
        'name': 'a·complexity^b·count^c + d·√count + e·complexity + f',
        'initial_guess': [1, 0.3, 1.4, 1, 1, 0]
    },
    'double_enhance_2': {
        'func': lambda complexity, count, a, b, c, d, e, f: a * (complexity ** b) * (count ** c) + d * np.sqrt(count) + e * np.log(count + 1) + f,
        'name': 'a·complexity^b·count^c + d·√count + e·ln(count+1) + f',
        'initial_guess': [1, 0.3, 1.4, 1, 1, 0]
    },
    'double_enhance_3': {
        'func': lambda complexity, count, a, b, c, d, e, f: a * (complexity ** b) * (count ** c) + d * complexity + e * np.sqrt(complexity) + f,
        'name': 'a·complexity^b·count^c + d·complexity + e·√complexity + f',
        'initial_guess': [1, 0.3, 1.4, 1, 0.5, 0]
    },
    
    # ========== CREATIVE/RANDOM EQUATIONS (avoiding local minima) ==========
    'creative_1': {
        'func': lambda complexity, count, a, b, c, d: a * (complexity ** 0.5) * (count ** 1.5) + b / (count + c) + d,
        'name': 'a·complexity^0.5·count^1.5 + b/(count+c) + d',
        'initial_guess': [1, 10, 10, 0]
    },
    'creative_2': {
        'func': lambda complexity, count, a, b, c, d: a * np.sqrt(complexity * count) * count**0.5 + b * count + c * complexity + d,
        'name': 'a·√(complexity·count)·count^0.5 + b·count + c·complexity + d',
        'initial_guess': [1, 0.5, 1, 0]
    },
    'creative_3': {
        'func': lambda complexity, count, a, b, c: a * (complexity * count) ** b + c * count,
        'name': 'a·(complexity·count)^b + c·count',
        'initial_guess': [1, 0.8, 0.5]
    },
    'creative_4': {
        'func': lambda complexity, count, a, b, c, d, e: a * count**b * (1 + c * complexity**d) + e,
        'name': 'a·count^b·(1 + c·complexity^d) + e',
        'initial_guess': [1, 1.5, 0.05, 0.5, 0]
    },
    'creative_5': {
        'func': lambda complexity, count, a, b, c, d: a * (count ** b) / (1 + c * np.exp(-complexity)) + d,
        'name': 'a·count^b/(1 + c·exp(-complexity)) + d',
        'initial_guess': [1, 1.5, 1, 0]
    },
    'creative_6': {
        'func': lambda complexity, count, a, b, c, d: a * (count ** b) * (1 + c / (1 + complexity)) + d,
        'name': 'a·count^b·(1 + c/(1+complexity)) + d',
        'initial_guess': [1, 1.5, 0.5, 0]
    },
    'creative_7': {
        'func': lambda complexity, count, a, b, c, d: a * (count ** b) + c * (complexity / (1 + complexity)) + d,
        'name': 'a·count^b + c·(complexity/(1+complexity)) + d',
        'initial_guess': [1, 1.5, 10, 0]
    },
    'creative_8': {
        'func': lambda complexity, count, a, b, c: a * np.exp(b * complexity) * count**c,
        'name': 'a·exp(b·complexity)·count^c',
        'initial_guess': [1, 0.01, 1.4]
    },
    'creative_9': {
        'func': lambda complexity, count, a, b, c, d: a * (count ** b) * np.tanh(c * complexity) + d,
        'name': 'a·count^b·tanh(c·complexity) + d',
        'initial_guess': [1, 1.5, 0.1, 0]
    },
    'creative_10': {
        'func': lambda complexity, count, a, b, c, d: a * (count ** b) * (1 - np.exp(-c * complexity)) + d,
        'name': 'a·count^b·(1 - exp(-c·complexity)) + d',
        'initial_guess': [1, 1.5, 0.1, 0]
    },
    
    # ========== RATIO-BASED APPROACHES ==========
    'ratio_simple_1': {
        'func': lambda complexity, count, a, b, c, d: a * (complexity / (complexity + b)) * (count ** c) + d,
        'name': 'a·(complexity/(complexity+b))·count^c + d',
        'initial_guess': [100, 1, 1.5, 0]
    },
    'ratio_simple_2': {
        'func': lambda complexity, count, a, b, c, d, e: a * (complexity / (complexity + b)) * (count ** c) + d * count + e,
        'name': 'a·(complexity/(complexity+b))·count^c + d·count + e',
        'initial_guess': [100, 1, 1.5, 0.5, 0]
    },
    'ratio_enhanced': {
        'func': lambda complexity, count, a, b, c, d, e, f: a * (complexity / (complexity + b)) * (count ** c) + d * count + e * np.sqrt(count) + f,
        'name': 'a·(complexity/(complexity+b))·count^c + d·count + e·√count + f',
        'initial_guess': [100, 1, 1.5, 0.5, 1, 0]
    },
    
    # ========== POLYNOMIAL HYBRID ==========
    'poly_hybrid_1': {
        'func': lambda complexity, count, a, b, c, d, e: a * (complexity ** b) * (count ** c) + d * count**2 + e,
        'name': 'a·complexity^b·count^c + d·count² + e',
        'initial_guess': [1, 0.3, 1.4, 0.0001, 0]
    },
    'poly_hybrid_2': {
        'func': lambda complexity, count, a, b, c, d, e: a * (complexity ** b) * (count ** c) + d * complexity**2 + e,
        'name': 'a·complexity^b·count^c + d·complexity² + e',
        'initial_guess': [1, 0.3, 1.4, 0.01, 0]
    },
    
    # ========== FRACTIONAL ROOT VARIATIONS ==========
    'frac_root_1': {
        'func': lambda complexity, count, a, b, c, d: a * (complexity ** b) * (count ** c) + d * (count ** 0.4),
        'name': 'a·complexity^b·count^c + d·count^0.4',
        'initial_guess': [1, 0.3, 1.4, 1]
    },
    'frac_root_2': {
        'func': lambda complexity, count, a, b, c, d, e: a * (complexity ** b) * (count ** c) + d * (count ** 0.6) + e,
        'name': 'a·complexity^b·count^c + d·count^0.6 + e',
        'initial_guess': [1, 0.3, 1.4, 1, 0]
    },
    'frac_root_3': {
        'func': lambda complexity, count, a, c, d, e: a * (complexity ** 0.3) * (count ** c) + d * np.sqrt(count) + e,
        'name': 'a·complexity^0.3·count^c + d·√count + e',
        'initial_guess': [1, 1.4, 1, 0]
    },
    
    # ========== EXTREME RANDOM EXPLORATION ==========
    'random_1': {
        'func': lambda complexity, count, a, b, c: a * (count ** 2.0) + b * complexity + c,
        'name': 'a·count² + b·complexity + c',
        'initial_guess': [0.001, 1, 0]
    },
    'random_2': {
        'func': lambda complexity, count, a, b, c, d: a * (count ** 1.8) + b * (complexity ** 0.4) + c * count + d,
        'name': 'a·count^1.8 + b·complexity^0.4 + c·count + d',
        'initial_guess': [0.01, 1, 0.5, 0]
    },
    'random_3': {
        'func': lambda complexity, count, a, b, c, d: a * np.log(count + 1) ** b + c * complexity + d,
        'name': 'a·ln(count+1)^b + c·complexity + d',
        'initial_guess': [1, 2, 1, 0]
    },
    'random_4': {
        'func': lambda complexity, count, a, b, c: a * np.exp(b * np.log(count)) + c,
        'name': 'a·exp(b·ln(count)) + c',
        'initial_guess': [1, 1.5, 0]
    },
    'random_5': {
        'func': lambda complexity, count, a, b, c, d: a * (complexity ** 0.1) * (count ** 2.0) + b * np.sqrt(count) + c * complexity + d,
        'name': 'a·complexity^0.1·count^2.0 + b·√count + c·complexity + d',
        'initial_guess': [0.1, 1, 1, 0]
    },
    'random_6': {
        'func': lambda complexity, count, a, b, c: a * (count ** 0.5) * (complexity ** 2.0) + b * count + c,
        'name': 'a·count^0.5·complexity^2.0 + b·count + c',
        'initial_guess': [0.1, 0.5, 0]
    },
    'random_7': {
        'func': lambda complexity, count, a, b, c, d: a * np.sin(b * complexity) * (count ** c) + d,
        'name': 'a·sin(b·complexity)·count^c + d',
        'initial_guess': [1, 0.1, 1.4, 0]
    },
    'random_8': {
        'func': lambda complexity, count, a, b, c, d: a * (count ** b) * np.cos(c * complexity) + d,
        'name': 'a·count^b·cos(c·complexity) + d',
        'initial_guess': [1, 1.5, 0.1, 0]
    },
    
    # ========== HYPER-SIMPLE SINGLE VARIABLE TESTS ==========
    'single_count_power': {
        'func': lambda complexity, count, a, b: a * (count ** b),
        'name': 'a·count^b',
        'initial_guess': [1, 1.5]
    },
    'single_complexity_power': {
        'func': lambda complexity, count, a, b: a * (complexity ** b),
        'name': 'a·complexity^b',
        'initial_guess': [1, 0.5]
    },
    'single_count_linear': {
        'func': lambda complexity, count, a, b: a * count + b,
        'name': 'a·count + b',
        'initial_guess': [1, 0]
    },
    'single_complexity_linear': {
        'func': lambda complexity, count, a, b: a * complexity + b,
        'name': 'a·complexity + b',
        'initial_guess': [1, 0]
    },
    
    # ========== CROSS-TERM EXPERIMENTS ==========
    'cross_term_1': {
        'func': lambda complexity, count, a, b, c: a * (complexity * count) ** b + c,
        'name': 'a·(complexity·count)^b + c',
        'initial_guess': [1, 0.8, 0]
    },
    'cross_term_2': {
        'func': lambda complexity, count, a, b, c, d: a * (complexity / count) ** b + c * count + d,
        'name': 'a·(complexity/count)^b + c·count + d',
        'initial_guess': [1, 0.5, 1, 0]
    },
    'cross_term_3': {
        'func': lambda complexity, count, a, b, c: a * (count / (complexity + 1e-10)) ** b + c,
        'name': 'a·(count/complexity)^b + c',
        'initial_guess': [1, 0.5, 0]
    },
    
    # ========== LOGARITHMIC VARIATIONS ==========
    'log_variations_1': {
        'func': lambda complexity, count, a, b, c, d: a * np.log(count + 1) * np.log(complexity + 1) + b * count + c * complexity + d,
        'name': 'a·ln(count+1)·ln(complexity+1) + b·count + c·complexity + d',
        'initial_guess': [1, 0.5, 1, 0]
    },
    'log_variations_2': {
        'func': lambda complexity, count, a, b, c: a * np.log(count + complexity + 1) + b * count + c,
        'name': 'a·ln(count+complexity+1) + b·count + c',
        'initial_guess': [1, 0.5, 0]
    },
    
    # ========== EXPONENTIAL DECAY MODELS ==========
    'exp_decay_1': {
        'func': lambda complexity, count, a, b, c, d: a * (1 - np.exp(-b * complexity)) * (count ** c) + d,
        'name': 'a·(1-exp(-b·complexity))·count^c + d',
        'initial_guess': [1, 0.1, 1.4, 0]
    },
    'exp_decay_2': {
        'func': lambda complexity, count, a, b, c, d: a * (count ** b) * (1 - np.exp(-c * complexity)) + d,
        'name': 'a·count^b·(1-exp(-c·complexity)) + d',
        'initial_guess': [1, 1.5, 0.1, 0]
    },
    
    # ========== SIGMOIDAL FUNCTIONS ==========
    'sigmoid_1': {
        'func': lambda complexity, count, a, b, c, d: a / (1 + np.exp(-b * complexity)) * (count ** c) + d,
        'name': 'a/(1+exp(-b·complexity))·count^c + d',
        'initial_guess': [100, 0.1, 1.4, 0]
    },
    'sigmoid_2': {
        'func': lambda complexity, count, a, b, c, d: a * (count ** b) / (1 + np.exp(-c * complexity)) + d,
        'name': 'a·count^b/(1+exp(-c·complexity)) + d',
        'initial_guess': [1, 1.5, 0.1, 0]
    },
    
    # ========== ITERATION 9: NEW SIMPLIFICATIONS & EXPLORATIONS ==========
    
    # Simplified versions of top performers (reduce from 7/6 params to 4-5)
    'simplified_top_1': {
        'func': lambda complexity, count, a, b, c, d: a * (complexity ** b) * (count ** c) + d,
        'name': 'a·complexity^b·count^c + d',
        'initial_guess': [1, 0.3, 1.4, 0]
    },
    'simplified_top_2': {
        'func': lambda complexity, count, a, b, c, d, e: a * (complexity ** b) * (count ** c) + d * count + e,
        'name': 'a·complexity^b·count^c + d·count + e',
        'initial_guess': [1, 0.3, 1.4, 0.5, 0]
    },
    'simplified_top_3': {
        'func': lambda complexity, count, a, b, c: a * (count ** b) * (1 - np.exp(-c * complexity)),
        'name': 'a·count^b·(1-exp(-c·complexity))',
        'initial_guess': [1, 1.5, 0.1]
    },
    'simplified_top_4': {
        'func': lambda complexity, count, a, b, c, d: a * (count ** b) + c * complexity + d,
        'name': 'a·count^b + c·complexity + d',
        'initial_guess': [1, 1.5, 1, 0]
    },
    
    # More aggressive count-only with various powers
    'count_power_1_3': {
        'func': lambda complexity, count, a, b: a * (count ** 1.3) + b,
        'name': 'a·count^1.3 + b',
        'initial_guess': [1, 0]
    },
    'count_power_1_4': {
        'func': lambda complexity, count, a, b: a * (count ** 1.4) + b,
        'name': 'a·count^1.4 + b',
        'initial_guess': [1, 0]
    },
    'count_power_1_6': {
        'func': lambda complexity, count, a, b: a * (count ** 1.6) + b,
        'name': 'a·count^1.6 + b',
        'initial_guess': [1, 0]
    },
    'count_power_1_7': {
        'func': lambda complexity, count, a, b: a * (count ** 1.7) + b,
        'name': 'a·count^1.7 + b',
        'initial_guess': [1, 0]
    },
    'count_power_1_8': {
        'func': lambda complexity, count, a, b: a * (count ** 1.8) + b,
        'name': 'a·count^1.8 + b',
        'initial_guess': [1, 0]
    },
    'count_power_1_9': {
        'func': lambda complexity, count, a, b: a * (count ** 1.9) + b,
        'name': 'a·count^1.9 + b',
        'initial_guess': [1, 0]
    },
    'count_power_2_0': {
        'func': lambda complexity, count, a, b: a * (count ** 2.0) + b,
        'name': 'a·count^2.0 + b',
        'initial_guess': [1, 0]
    },
    
    # Complexity-only with various transformations
    'complexity_sqrt': {
        'func': lambda complexity, count, a, b: a * np.sqrt(complexity) + b,
        'name': 'a·√complexity + b',
        'initial_guess': [1, 0]
    },
    'complexity_log': {
        'func': lambda complexity, count, a, b: a * np.log(complexity + 1) + b,
        'name': 'a·ln(complexity+1) + b',
        'initial_guess': [1, 0]
    },
    'complexity_cube': {
        'func': lambda complexity, count, a, b: a * (complexity ** 3) + b,
        'name': 'a·complexity³ + b',
        'initial_guess': [1, 0]
    },
    'complexity_power_04': {
        'func': lambda complexity, count, a, b: a * (complexity ** 0.4) + b,
        'name': 'a·complexity^0.4 + b',
        'initial_guess': [1, 0]
    },
    'complexity_power_06': {
        'func': lambda complexity, count, a, b: a * (complexity ** 0.6) + b,
        'name': 'a·complexity^0.6 + b',
        'initial_guess': [1, 0]
    },
    
    # Ratio models simplified
    'ratio_ultra_simple': {
        'func': lambda complexity, count, a, b, c: a * (complexity / (complexity + 1)) * (count ** b) + c,
        'name': 'a·(complexity/(complexity+1))·count^b + c',
        'initial_guess': [100, 1.5, 0]
    },
    'ratio_with_linear': {
        'func': lambda complexity, count, a, b, c, d: a * (complexity / (complexity + 1)) * (count ** b) + c * count + d,
        'name': 'a·(complexity/(complexity+1))·count^b + c·count + d',
        'initial_guess': [100, 1.5, 0.5, 0]
    },
    
    # Product terms with various combinations
    'product_simple_1': {
        'func': lambda complexity, count, a, b, c: a * complexity * count + b * count + c,
        'name': 'a·complexity·count + b·count + c',
        'initial_guess': [1, 1, 0]
    },
    'product_simple_2': {
        'func': lambda complexity, count, a, b, c: a * np.sqrt(complexity * count) + b * count + c,
        'name': 'a·√(complexity·count) + b·count + c',
        'initial_guess': [1, 1, 0]
    },
    'product_power': {
        'func': lambda complexity, count, a, b, c: a * (complexity * count) ** b + c,
        'name': 'a·(complexity·count)^b + c',
        'initial_guess': [1, 0.7, 0]
    },
    
    # Weighted sums with fixed exponents
    'weighted_fixed_1': {
        'func': lambda complexity, count, a, b, c: a * (complexity ** 0.5) + b * (count ** 1.5) + c,
        'name': 'a·complexity^0.5 + b·count^1.5 + c',
        'initial_guess': [1, 1, 0]
    },
    'weighted_fixed_2': {
        'func': lambda complexity, count, a, b, c: a * (complexity ** 0.3) + b * (count ** 1.6) + c,
        'name': 'a·complexity^0.3 + b·count^1.6 + c',
        'initial_guess': [1, 1, 0]
    },
    'weighted_fixed_3': {
        'func': lambda complexity, count, a, b, c, d: a * complexity + b * (count ** 1.5) + c * count + d,
        'name': 'a·complexity + b·count^1.5 + c·count + d',
        'initial_guess': [1, 1, 0.5, 0]
    },
    
    # Logarithmic combinations
    'log_combo_1': {
        'func': lambda complexity, count, a, b, c: a * np.log(count + 1) * (count ** b) + c,
        'name': 'a·ln(count+1)·count^b + c',
        'initial_guess': [1, 1.0, 0]
    },
    'log_combo_2': {
        'func': lambda complexity, count, a, b, c, d: a * np.log(count + 1) + b * (count ** c) + d,
        'name': 'a·ln(count+1) + b·count^c + d',
        'initial_guess': [1, 1, 1.5, 0]
    },
    'log_combo_3': {
        'func': lambda complexity, count, a, b, c: a * np.log(complexity + 1) + b * (count ** 1.5) + c,
        'name': 'a·ln(complexity+1) + b·count^1.5 + c',
        'initial_guess': [1, 1, 0]
    },
    
    # Polynomial with interaction
    'poly_interaction_1': {
        'func': lambda complexity, count, a, b, c, d: a * count + b * complexity + c * complexity * count + d,
        'name': 'a·count + b·complexity + c·complexity·count + d',
        'initial_guess': [1, 1, 0.01, 0]
    },
    'poly_interaction_2': {
        'func': lambda complexity, count, a, b, c, d, e: a * (count ** 2) + b * complexity + c * complexity * count + d * count + e,
        'name': 'a·count² + b·complexity + c·complexity·count + d·count + e',
        'initial_guess': [0.001, 1, 0.01, 1, 0]
    },
    
    # Exponential with count
    'exp_count_1': {
        'func': lambda complexity, count, a, b, c: a * count * np.exp(b * complexity) + c,
        'name': 'a·count·exp(b·complexity) + c',
        'initial_guess': [1, 0.001, 0]
    },
    'exp_count_2': {
        'func': lambda complexity, count, a, b, c, d: a * (count ** b) * np.exp(c * complexity) + d,
        'name': 'a·count^b·exp(c·complexity) + d',
        'initial_guess': [1, 1.5, 0.001, 0]
    },
    
    # Hyperbolic functions
    'tanh_simple': {
        'func': lambda complexity, count, a, b, c: a * (count ** b) * np.tanh(complexity) + c,
        'name': 'a·count^b·tanh(complexity) + c',
        'initial_guess': [1, 1.5, 0]
    },
    'sinh_combo': {
        'func': lambda complexity, count, a, b, c, d: a * (count ** b) + c * np.sinh(complexity * 0.1) + d,
        'name': 'a·count^b + c·sinh(0.1·complexity) + d',
        'initial_guess': [1, 1.5, 1, 0]
    },
    
    # Piecewise-like smooth functions
    'smooth_step_1': {
        'func': lambda complexity, count, a, b, c, d: a * (count ** b) * (complexity / (1 + complexity)) + c * count + d,
        'name': 'a·count^b·(complexity/(1+complexity)) + c·count + d',
        'initial_guess': [1, 1.5, 0.5, 0]
    },
    'smooth_step_2': {
        'func': lambda complexity, count, a, b, c: a * (count ** b) / (1 + complexity) + c,
        'name': 'a·count^b/(1+complexity) + c',
        'initial_guess': [1, 1.5, 0]
    },
    
    # Random explorations with unusual patterns
    'random_new_1': {
        'func': lambda complexity, count, a, b, c: a * (count ** 1.55) + b * (complexity ** 0.33) + c,
        'name': 'a·count^1.55 + b·complexity^0.33 + c',
        'initial_guess': [1, 1, 0]
    },
    'random_new_2': {
        'func': lambda complexity, count, a, b, c: a * (count ** 1.62) * (1 + b * np.log(complexity + 1)) + c,
        'name': 'a·count^1.62·(1 + b·ln(complexity+1)) + c',
        'initial_guess': [1, 0.1, 0]
    },
    'random_new_3': {
        'func': lambda complexity, count, a, b, c, d: a * (count ** 1.45) + b * np.sqrt(complexity) + c * np.sqrt(count) + d,
        'name': 'a·count^1.45 + b·√complexity + c·√count + d',
        'initial_guess': [1, 1, 1, 0]
    },
    'random_new_4': {
        'func': lambda complexity, count, a, b, c: a * (count + complexity) ** b + c,
        'name': 'a·(count+complexity)^b + c',
        'initial_guess': [1, 1.2, 0]
    },
    'random_new_5': {
        'func': lambda complexity, count, a, b, c: a * count * (count ** b) / (1 + complexity) + c,
        'name': 'a·count·count^b/(1+complexity) + c',
        'initial_guess': [1, 0.5, 0]
    },
    
    # Ultra-simple 2-parameter models
    'ultra_simple_1': {
        'func': lambda complexity, count, a: a * (count ** 1.55),
        'name': 'a·count^1.55',
        'initial_guess': [1]
    },
    'ultra_simple_2': {
        'func': lambda complexity, count, a: a * count * np.sqrt(count),
        'name': 'a·count^1.5',
        'initial_guess': [1]
    },
    'ultra_simple_3': {
        'func': lambda complexity, count, a: a * (count ** 1.4),
        'name': 'a·count^1.4',
        'initial_guess': [1]
    },
    'ultra_simple_4': {
        'func': lambda complexity, count, a: a * (count ** 1.6),
        'name': 'a·count^1.6',
        'initial_guess': [1]
    },
    
    # ========== ITERATION 10: TARGET 0.6 - REFINE TOP PATTERNS ==========
    # Best so far: 0.558 with optimal_7param_1
    # Analysis: count^c dominates, complexity adds small correction
    # Strategy: Simplify optimal_7param_1, explore count-only more, test intermediate powers
    
    # Simplified versions of optimal_7param_1 (reduce from 7 to 3-5 params)
    'optimal_simplified_1': {
        'func': lambda complexity, count, a, b, c: a * (count ** b) + c * np.sqrt(complexity),
        'name': 'a·count^b + c·√complexity',
        'initial_guess': [1, 1.4, 1]
    },
    'optimal_simplified_2': {
        'func': lambda complexity, count, a, b, c, d: a * (count ** b) + c * complexity + d,
        'name': 'a·count^b + c·complexity + d',
        'initial_guess': [1, 1.4, 1, 0]
    },
    'optimal_simplified_3': {
        'func': lambda complexity, count, a, b, c, d: a * (count ** b) + c * np.sqrt(complexity) + d,
        'name': 'a·count^b + c·√complexity + d',
        'initial_guess': [1, 1.4, 1, 0]
    },
    'optimal_simplified_4': {
        'func': lambda complexity, count, a, b, c, d, e: a * (count ** b) + c * complexity + d * np.sqrt(complexity) + e,
        'name': 'a·count^b + c·complexity + d·√complexity + e',
        'initial_guess': [1, 1.4, 1, 1, 0]
    },
    
    # Fine-tune count power around 1.4-1.5 range (seems optimal)
    'count_fine_1_42': {
        'func': lambda complexity, count, a, b: a * (count ** 1.42) + b,
        'name': 'a·count^1.42 + b',
        'initial_guess': [1, 0]
    },
    'count_fine_1_44': {
        'func': lambda complexity, count, a, b: a * (count ** 1.44) + b,
        'name': 'a·count^1.44 + b',
        'initial_guess': [1, 0]
    },
    'count_fine_1_46': {
        'func': lambda complexity, count, a, b: a * (count ** 1.46) + b,
        'name': 'a·count^1.46 + b',
        'initial_guess': [1, 0]
    },
    'count_fine_1_48': {
        'func': lambda complexity, count, a, b: a * (count ** 1.48) + b,
        'name': 'a·count^1.48 + b',
        'initial_guess': [1, 0]
    },
    'count_fine_1_52': {
        'func': lambda complexity, count, a, b: a * (count ** 1.52) + b,
        'name': 'a·count^1.52 + b',
        'initial_guess': [1, 0]
    },
    'count_fine_1_54': {
        'func': lambda complexity, count, a, b: a * (count ** 1.54) + b,
        'name': 'a·count^1.54 + b',
        'initial_guess': [1, 0]
    },
    'count_fine_1_56': {
        'func': lambda complexity, count, a, b: a * (count ** 1.56) + b,
        'name': 'a·count^1.56 + b',
        'initial_guess': [1, 0]
    },
    'count_fine_1_58': {
        'func': lambda complexity, count, a, b: a * (count ** 1.58) + b,
        'name': 'a·count^1.58 + b',
        'initial_guess': [1, 0]
    },
    
    # Add complexity as small correction to count-power
    'count_complexity_1': {
        'func': lambda complexity, count, a, b: a * (count ** 1.42) + b * complexity,
        'name': 'a·count^1.42 + b·complexity',
        'initial_guess': [1, 1]
    },
    'count_complexity_2': {
        'func': lambda complexity, count, a, b: a * (count ** 1.5) + b * np.sqrt(complexity),
        'name': 'a·count^1.5 + b·√complexity',
        'initial_guess': [1, 1]
    },
    'count_complexity_3': {
        'func': lambda complexity, count, a, b: a * (count ** 1.6) + b * np.log(complexity + 1),
        'name': 'a·count^1.6 + b·ln(complexity+1)',
        'initial_guess': [1, 1]
    },
    'count_complexity_4': {
        'func': lambda complexity, count, a, b, c: a * (count ** 1.45) + b * complexity + c,
        'name': 'a·count^1.45 + b·complexity + c',
        'initial_guess': [1, 1, 0]
    },
    'count_complexity_5': {
        'func': lambda complexity, count, a, b, c: a * (count ** 1.55) + b * np.sqrt(complexity) + c,
        'name': 'a·count^1.55 + b·√complexity + c',
        'initial_guess': [1, 1, 0]
    },
    
    # Ratio-based with optimal count power
    'ratio_optimal_1': {
        'func': lambda complexity, count, a, b: a * (complexity / (complexity + 10)) * (count ** 1.4) + b,
        'name': 'a·(complexity/(complexity+10))·count^1.4 + b',
        'initial_guess': [100, 0]
    },
    'ratio_optimal_2': {
        'func': lambda complexity, count, a, b, c: a * (complexity / (complexity + b)) * (count ** 1.5) + c,
        'name': 'a·(complexity/(complexity+b))·count^1.5 + c',
        'initial_guess': [100, 10, 0]
    },
    'ratio_optimal_3': {
        'func': lambda complexity, count, a, b, c: a * (complexity / (complexity + 1)) * (count ** b) + c,
        'name': 'a·(complexity/(complexity+1))·count^b + c',
        'initial_guess': [100, 1.4, 0]
    },
    
    # Explore interaction between complexity^small and count^large
    'interaction_1': {
        'func': lambda complexity, count, a, b: a * (complexity ** 0.1) * (count ** 1.5) + b,
        'name': 'a·complexity^0.1·count^1.5 + b',
        'initial_guess': [1, 0]
    },
    'interaction_2': {
        'func': lambda complexity, count, a, b: a * (complexity ** 0.2) * (count ** 1.4) + b,
        'name': 'a·complexity^0.2·count^1.4 + b',
        'initial_guess': [1, 0]
    },
    'interaction_3': {
        'func': lambda complexity, count, a, b: a * (complexity ** 0.3) * (count ** 1.3) + b,
        'name': 'a·complexity^0.3·count^1.3 + b',
        'initial_guess': [1, 0]
    },
    'interaction_4': {
        'func': lambda complexity, count, a, b, c: a * (complexity ** 0.15) * (count ** b) + c,
        'name': 'a·complexity^0.15·count^b + c',
        'initial_guess': [1, 1.5, 0]
    },
    'interaction_5': {
        'func': lambda complexity, count, a, b, c: a * (complexity ** b) * (count ** 1.5) + c,
        'name': 'a·complexity^b·count^1.5 + c',
        'initial_guess': [1, 0.2, 0]
    },
    
    # Multiplicative corrections to count-power
    'multiplicative_1': {
        'func': lambda complexity, count, a, b: a * (count ** 1.45) * (1 + b * complexity),
        'name': 'a·count^1.45·(1 + b·complexity)',
        'initial_guess': [1, 0.001]
    },
    'multiplicative_2': {
        'func': lambda complexity, count, a, b: a * (count ** 1.5) * (1 + b * np.sqrt(complexity)),
        'name': 'a·count^1.5·(1 + b·√complexity)',
        'initial_guess': [1, 0.01]
    },
    'multiplicative_3': {
        'func': lambda complexity, count, a, b: a * (count ** 1.5) * (1 + b * np.log(complexity + 1)),
        'name': 'a·count^1.5·(1 + b·ln(complexity+1))',
        'initial_guess': [1, 0.01]
    },
    
    # Test if purely count-based with offset works better
    'count_pure_offset_1': {
        'func': lambda complexity, count, a, b, c: a * (count ** b) + c,
        'name': 'a·count^b + c (pure)',
        'initial_guess': [1, 1.42, 10]
    },
    'count_pure_offset_2': {
        'func': lambda complexity, count, a, b, c, d: a * (count ** b) + c * np.sqrt(count) + d,
        'name': 'a·count^b + c·√count + d (pure)',
        'initial_guess': [1, 1.2, 1, 10]
    },
    
    # Weighted combination with fixed powers
    'weighted_optimal_1': {
        'func': lambda complexity, count, a, b, c: a * (count ** 1.45) + b * (complexity ** 0.5) + c,
        'name': 'a·count^1.45 + b·complexity^0.5 + c',
        'initial_guess': [1, 1, 10]
    },
    'weighted_optimal_2': {
        'func': lambda complexity, count, a, b, c: a * (count ** 1.5) + b * (complexity ** 0.3) + c,
        'name': 'a·count^1.5 + b·complexity^0.3 + c',
        'initial_guess': [1, 1, 10]
    },
    'weighted_optimal_3': {
        'func': lambda complexity, count, a, b, c: a * (count ** 1.55) + b * (complexity ** 0.25) + c,
        'name': 'a·count^1.55 + b·complexity^0.25 + c',
        'initial_guess': [1, 1, 10]
    },
    
    # Random exploration to avoid local minimum
    'random_explore_1': {
        'func': lambda complexity, count, a, b: a * (count ** 1.63) + b,
        'name': 'a·count^1.63 + b',
        'initial_guess': [1, 10]
    },
    'random_explore_2': {
        'func': lambda complexity, count, a, b: a * (count ** 1.37) + b,
        'name': 'a·count^1.37 + b',
        'initial_guess': [1, 10]
    },
    'random_explore_3': {
        'func': lambda complexity, count, a, b, c: a * (count ** 1.47) + b * (complexity ** 0.35) + c,
        'name': 'a·count^1.47 + b·complexity^0.35 + c',
        'initial_guess': [1, 1, 10]
    },
    'random_explore_4': {
        'func': lambda complexity, count, a, b, c: a * (count ** 1.53) + b * np.log(complexity + 2) + c,
        'name': 'a·count^1.53 + b·ln(complexity+2) + c',
        'initial_guess': [1, 1, 10]
    },
    'random_explore_5': {
        'func': lambda complexity, count, a, b, c: a * (count ** 1.67) * (1 + b / (complexity + 1)) + c,
        'name': 'a·count^1.67·(1 + b/(complexity+1)) + c',
        'initial_guess': [1, 0.1, 10]
    },
    
    # ========== ITERATION 11: ULTRA-FOCUSED ON 0.6 TARGET ==========
    # Best: 0.558 (still can't break 0.6)
    # Strategy: Test if pure count models can be optimized further
    # Focus on count^b + small_correction pattern
    
    # Super simple: just optimize count power
    'ultra_count_opt_1': {
        'func': lambda complexity, count, a, b, c: a * (count ** 1.405) + b * complexity ** 0.5 + c,
        'name': 'a·count^1.405 + b·complexity^0.5 + c',
        'initial_guess': [1, 1, 10]
    },
    'ultra_count_opt_2': {
        'func': lambda complexity, count, a, b, c: a * (count ** 1.395) + b * complexity ** 0.5 + c,
        'name': 'a·count^1.395 + b·complexity^0.5 + c',
        'initial_guess': [1, 1, 10]
    },
    'ultra_count_opt_3': {
        'func': lambda complexity, count, a, b, c: a * (count ** 1.415) + b * complexity ** 0.5 + c,
        'name': 'a·count^1.415 + b·complexity^0.5 + c',
        'initial_guess': [1, 1, 10]
    },
    'ultra_count_opt_4': {
        'func': lambda complexity, count, a, b, c: a * (count ** 1.425) + b * complexity ** 0.5 + c,
        'name': 'a·count^1.425 + b·complexity^0.5 + c',
        'initial_guess': [1, 1, 10]
    },
    'ultra_count_opt_5': {
        'func': lambda complexity, count, a, b, c: a * (count ** 1.435) + b * complexity ** 0.5 + c,
        'name': 'a·count^1.435 + b·complexity^0.5 + c',
        'initial_guess': [1, 1, 10]
    },
    
    # Test hybrid approach: let count power optimize with small complexity term
    'hybrid_opt_1': {
        'func': lambda complexity, count, a, b, c, d: a * (count ** b) + c * (complexity ** 0.2) + d,
        'name': 'a·count^b + c·complexity^0.2 + d',
        'initial_guess': [1, 1.4, 1, 10]
    },
    'hybrid_opt_2': {
        'func': lambda complexity, count, a, b, c, d: a * (count ** b) + c * (complexity ** 0.3) + d,
        'name': 'a·count^b + c·complexity^0.3 + d',
        'initial_guess': [1, 1.4, 1, 10]
    },
    'hybrid_opt_3': {
        'func': lambda complexity, count, a, b, c, d: a * (count ** b) + c * (complexity ** 0.4) + d,
        'name': 'a·count^b + c·complexity^0.4 + d',
        'initial_guess': [1, 1.4, 1, 10]
    },
    'hybrid_opt_4': {
        'func': lambda complexity, count, a, b, c, d: a * (count ** b) + c * (complexity ** 0.5) + d,
        'name': 'a·count^b + c·complexity^0.5 + d',
        'initial_guess': [1, 1.4, 1, 10]
    },
    'hybrid_opt_5': {
        'func': lambda complexity, count, a, b, c, d: a * (count ** b) + c * np.log(complexity + 1) + d,
        'name': 'a·count^b + c·ln(complexity+1) + d',
        'initial_guess': [1, 1.4, 1, 10]
    },
    
    # Try non-linear offsets
    'nonlinear_offset_1': {
        'func': lambda complexity, count, a, b, c, d: a * (count ** 1.42) + b * np.sqrt(complexity) + c * np.sqrt(count) + d,
        'name': 'a·count^1.42 + b·√complexity + c·√count + d',
        'initial_guess': [1, 1, 1, 10]
    },
    'nonlinear_offset_2': {
        'func': lambda complexity, count, a, b, c, d: a * (count ** 1.5) + b * np.log(complexity + 1) + c * np.log(count + 1) + d,
        'name': 'a·count^1.5 + b·ln(complexity+1) + c·ln(count+1) + d',
        'initial_guess': [1, 1, 1, 10]
    },
    
    # Minimal parameter versions - try to beat 0.558 with 3 params
    'minimal_3param_1': {
        'func': lambda complexity, count, a, b, c: a * (count ** 1.42) + b * complexity + c,
        'name': 'a·count^1.42 + b·complexity + c [minimal]',
        'initial_guess': [1, 1, 10]
    },
    'minimal_3param_2': {
        'func': lambda complexity, count, a, b, c: a * (count ** 1.45) + b * np.sqrt(complexity) + c,
        'name': 'a·count^1.45 + b·√complexity + c [minimal]',
        'initial_guess': [1, 1, 10]
    },
    'minimal_3param_3': {
        'func': lambda complexity, count, a, b, c: a * (count ** 1.48) + b * np.log(complexity + 1) + c,
        'name': 'a·count^1.48 + b·ln(complexity+1) + c [minimal]',
        'initial_guess': [1, 1, 10]
    },
    'minimal_3param_4': {
        'func': lambda complexity, count, a, b, c: a * (count ** 1.5) + b * complexity ** 0.3 + c,
        'name': 'a·count^1.5 + b·complexity^0.3 + c [minimal]',
        'initial_guess': [1, 1, 10]
    },
    'minimal_3param_5': {
        'func': lambda complexity, count, a, b, c: a * (count ** 1.55) + b * complexity ** 0.2 + c,
        'name': 'a·count^1.55 + b·complexity^0.2 + c [minimal]',
        'initial_guess': [1, 1, 10]
    },
    
    # Try count^1.4 exactly with various complexity corrections
    'count_1_4_var_1': {
        'func': lambda complexity, count, a, b, c: a * (count ** 1.4) + b * complexity + c,
        'name': 'a·count^1.4 + b·complexity + c [fixed]',
        'initial_guess': [1, 1, 10]
    },
    'count_1_4_var_2': {
        'func': lambda complexity, count, a, b, c: a * (count ** 1.4) + b * np.sqrt(complexity) + c,
        'name': 'a·count^1.4 + b·√complexity + c [fixed]',
        'initial_guess': [1, 1, 10]
    },
    'count_1_4_var_3': {
        'func': lambda complexity, count, a, b, c: a * (count ** 1.4) + b * np.log(complexity + 1) + c,
        'name': 'a·count^1.4 + b·ln(complexity+1) + c [fixed]',
        'initial_guess': [1, 1, 10]
    },
    'count_1_4_var_4': {
        'func': lambda complexity, count, a, b, c: a * (count ** 1.4) + b * (complexity ** 0.25) + c,
        'name': 'a·count^1.4 + b·complexity^0.25 + c [fixed]',
        'initial_guess': [1, 1, 10]
    },
    'count_1_4_var_5': {
        'func': lambda complexity, count, a, b, c: a * (count ** 1.4) + b * (complexity ** 0.33) + c,
        'name': 'a·count^1.4 + b·complexity^0.33 + c [fixed]',
        'initial_guess': [1, 1, 10]
    },
    
    # More aggressive random search
    'aggressive_random_1': {
        'func': lambda complexity, count, a, b, c: a * (count ** 1.618) + b * (complexity ** 0.382) + c,
        'name': 'a·count^1.618 + b·complexity^0.382 + c [golden]',
        'initial_guess': [1, 1, 10]
    },
    'aggressive_random_2': {
        'func': lambda complexity, count, a, b, c: a * (count ** 1.732) + b / (complexity + 1) + c,
        'name': 'a·count^1.732 + b/(complexity+1) + c [sqrt2]',
        'initial_guess': [1, 1, 10]
    },
    'aggressive_random_3': {
        'func': lambda complexity, count, a, b, c: a * (count ** 1.414) + b * np.arctan(complexity) + c,
        'name': 'a·count^1.414 + b·arctan(complexity) + c [sqrt2]',
        'initial_guess': [1, 1, 10]
    },
}