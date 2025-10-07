"""
Evolutionary Equation Discovery System

This module implements an iterative genetic algorithm that evolves equations
to predict benchmark scores. It loads equations, evaluates them, selects the
best ones, and uses genetic algorithms to generate improved variations.
"""

import numpy as np
import pandas as pd
import warnings
from scipy.optimize import curve_fit
from typing import List, Dict, Tuple, Any
import os
import json
import time
from datetime import datetime

import ga_building_blocks as gbb
import ga_engine as gae

# ============================================================================
# EQUATION FILE I/O
# ============================================================================

def equations_to_dict(equations: List[gae.Equation], allowed_vars: List[str] = None, 
                     debug: bool = False) -> Tuple[Dict[str, Dict], Dict[str, int]]:
    """
    Convert list of Equation objects to dictionary format for z_eqs_to_test.py
    
    Args:
        equations: List of Equation objects to convert
        allowed_vars: List of allowed variable names (default: ['complexity', 'count'])
        debug: Whether to print debug information
    
    Returns:
        - Dictionary of equation_id -> equation_info
        - Dictionary of conversion failure reasons and counts
    """
    
    if allowed_vars is None:
        allowed_vars = ['complexity', 'count']
    
    result = {}
    conversion_failures = {
        'lambda_creation_error': 0,
        'to_string_error': 0,
        'metadata_error': 0,
        'other_error': 0,
        'no_tree': 0,
        'tree_is_none': 0,
        'missing_attributes': 0,
        'missing_param_guess': 0,
        'duplicate_id': 0  # Track duplicate unique_ids
    }
    
    seen_ids = set()  # Track which eq_ids we've seen
    
    # Track first few failures for debugging
    failure_samples = []
    max_samples = 5
    
    for idx, eq in enumerate(equations):
        # Check if equation itself is None
        if eq is None:
            conversion_failures['other_error'] += 1
            if debug and len(failure_samples) < max_samples:
                failure_samples.append(f"Equation {idx}: equation object is None")
            continue
        
        try:
            # Validate equation has required attributes
            if not hasattr(eq, 'generation') or not hasattr(eq, 'unique_id'):
                conversion_failures['missing_attributes'] += 1
                if debug and len(failure_samples) < max_samples:
                    failure_samples.append(f"Equation {idx}: missing generation or unique_id")
                continue
            
            # Use unique_id instead of index to prevent mapping issues after sorting
            eq_id = f"ga_gen{eq.generation}_uid{eq.unique_id}"
            
            # Check for duplicate IDs
            if eq_id in seen_ids:
                conversion_failures['duplicate_id'] += 1
                if debug and len(failure_samples) < max_samples:
                    failure_samples.append(f"Equation {idx}: duplicate unique_id {eq.unique_id} (eq_id: {eq_id})")
                continue
            seen_ids.add(eq_id)
            
            # Validate equation has tree for lambda creation
            if not hasattr(eq, 'tree'):
                conversion_failures['no_tree'] += 1
                if debug and len(failure_samples) < max_samples:
                    failure_samples.append(f"{eq_id}: has no 'tree' attribute")
                continue
            
            if eq.tree is None:
                conversion_failures['tree_is_none'] += 1
                if debug and len(failure_samples) < max_samples:
                    failure_samples.append(f"{eq_id}: tree is None")
                continue
            
            # Create the lambda function - this can fail if tree has unknown operations
            try:
                func = eq.to_lambda(allowed_vars=allowed_vars)
                if func is None:
                    conversion_failures['lambda_creation_error'] += 1
                    if debug and len(failure_samples) < max_samples:
                        failure_samples.append(f"{eq_id}: to_lambda() returned None")
                    continue
            except Exception as e:
                conversion_failures['lambda_creation_error'] += 1
                if debug and len(failure_samples) < max_samples:
                    failure_samples.append(f"{eq_id}: to_lambda() raised {type(e).__name__}: {e}")
                continue
            
            # Create human-readable name
            try:
                name = eq.to_string()
            except Exception as e:
                conversion_failures['to_string_error'] += 1
                if debug:
                    print(f"    DEBUG: Failed to_string for {eq_id}: {type(e).__name__}: {e}")
                name = f"[Error: {type(e).__name__}]"
            
            # Get initial guess - validate it exists
            if not hasattr(eq, 'param_initial_guess'):
                conversion_failures['missing_param_guess'] += 1
                if debug and len(failure_samples) < max_samples:
                    failure_samples.append(f"{eq_id}: missing param_initial_guess attribute")
                continue
            initial_guess = eq.param_initial_guess
            
            # Build metadata
            try:
                metadata = {
                    'generation': eq.generation,
                    'fitness': float(eq.fitness),
                    'avg_correlation': float(eq.avg_correlation),
                    'avg_r2': float(eq.avg_r2),
                    'complexity': float(eq.get_complexity()),
                    'depth': eq.get_depth(),
                    'size': eq.get_size(),
                    'parent_ids': eq.parent_ids
                }
            except Exception as e:
                conversion_failures['metadata_error'] += 1
                if debug and len(failure_samples) < max_samples:
                    failure_samples.append(f"{eq_id}: metadata error {type(e).__name__}: {e}")
                # Use minimal metadata
                metadata = {
                    'generation': eq.generation,
                    'fitness': float(eq.fitness) if hasattr(eq, 'fitness') else -1.0,
                    'avg_correlation': 0.0,
                    'avg_r2': 0.0,
                    'complexity': 0.0,
                    'depth': 0,
                    'size': 0,
                    'parent_ids': []
                }
            
            result[eq_id] = {
                'func': func,
                'name': name,
                'initial_guess': initial_guess,
                'equation_object': eq,  # Store reference to original equation
                'metadata': metadata
            }
            
        except Exception as e:
            conversion_failures['other_error'] += 1
            if debug and len(failure_samples) < max_samples:
                failure_samples.append(f"Equation {idx}: unexpected {type(e).__name__}: {e}")
            continue
    
    # Print failure samples if debug mode
    if debug and failure_samples:
        print(f"  Conversion failure samples ({len(failure_samples)}):")
        for sample in failure_samples:
            print(f"    • {sample}")
    
    return result, conversion_failures

def save_equations_to_file(equations: List[gae.Equation], filepath: str):
    """Save equations to a Python file in the format expected by d_getKnowledge.py"""
    
    import json
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('"""\n')
        f.write('Auto-generated equations from genetic algorithm\n')
        f.write(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write('"""\n\n')
        f.write('import numpy as np\n\n')
        
        # Save tree structures as JSON for checkpoint resume
        f.write('# Tree structures for checkpoint resume (JSON format)\n')
        f.write('TREE_STRUCTURES = [\n')
        for eq in equations:
            tree_dict = eq.tree.to_dict()
            tree_json = json.dumps(tree_dict)
            f.write(f'    {tree_json},\n')
        f.write(']\n\n')
        
        # Save metadata for checkpoint resume
        f.write('# Equation metadata for checkpoint resume\n')
        f.write('EQUATION_METADATA = [\n')
        for eq in equations:
            # Convert numpy types and handle infinity values
            metadata = {
                'num_params': int(eq.num_params),
                'param_initial_guess': [float(x) for x in eq.param_initial_guess],
                'fitness': float(eq.fitness) if not np.isinf(eq.fitness) else (1e308 if eq.fitness > 0 else -1e308),
                'avg_correlation': float(eq.avg_correlation) if not np.isinf(eq.avg_correlation) else 0.0,
                'avg_r2': float(eq.avg_r2) if not np.isinf(eq.avg_r2) else 0.0,
                'simplicity_score': float(eq.simplicity_score),
                'generation': int(eq.generation)
            }
            metadata_json = json.dumps(metadata)
            f.write(f'    {metadata_json},\n')
        f.write(']\n\n')
        
        f.write('EQUATIONS_TO_TEST = {\n')
        
        for idx, eq in enumerate(equations):
            eq_id = f"ga_gen{eq.generation}_eq{idx}"
            
            # Write equation entry
            f.write(f'    # Generation {eq.generation}, Fitness: {eq.fitness:.6f}, Correlation: {eq.avg_correlation:.6f}\n')
            f.write(f'    \'{eq_id}\': {{\n')
            
            # Write the function
            # We need to convert the tree to a lambda string
            func_str = tree_to_lambda_string(eq.tree)
            f.write(f'        \'func\': lambda complexity, count, {", ".join([f"p{i}" for i in range(eq.num_params)])}: {func_str},\n')
            
            # Write name
            name = eq.to_string()
            if len(name) > 100:
                name = name[:97] + "..."
            f.write(f'        \'name\': \'{name}\',\n')
            
            # Write initial guess
            guess_str = "[" + ", ".join([f"{x:.4f}" for x in eq.param_initial_guess]) + "]"
            f.write(f'        \'initial_guess\': {guess_str}\n')
            
            f.write(f'    }},\n')
        
        f.write('}\n')

def tree_to_lambda_string(node: gae.ExprNode) -> str:
    """Convert expression tree to lambda string for file output"""
    
    if node.op == 'complexity':
        return 'complexity'
    elif node.op == 'count':
        return 'count'
    elif node.op == 'param':
        return f'p{node.param_idx}'
    else:
        op_obj = gbb.ALL_OPS.get(node.op)
        if not op_obj:
            return 'np.nan'
        
        if op_obj.arity == 0:
            return node.op
        elif op_obj.arity == 1:
            child_str = tree_to_lambda_string(node.children[0])
            
            if node.op == 'sqrt':
                return f'np.sqrt(np.abs({child_str}))'
            elif node.op == 'log':
                return f'np.log(np.abs({child_str}) + 1)'
            elif node.op == 'log2':
                return f'np.log2(np.abs({child_str}) + 1)'
            elif node.op == 'log10':
                return f'np.log10(np.abs({child_str}) + 1)'
            elif node.op == 'exp':
                return f'np.exp(np.clip({child_str}, -100, 100))'
            elif node.op == 'sin':
                return f'np.sin({child_str})'
            elif node.op == 'cos':
                return f'np.cos({child_str})'
            elif node.op == 'tan':
                return f'np.tan(np.clip({child_str}, -np.pi/2 + 0.1, np.pi/2 - 0.1))'
            elif node.op == 'abs':
                return f'np.abs({child_str})'
            elif node.op == 'square':
                return f'({child_str})**2'
            elif node.op == 'cube':
                return f'({child_str})**3'
            elif node.op == 'neg':
                return f'-({child_str})'
            elif node.op == 'reciprocal':
                return f'(1.0 / (np.abs({child_str}) + 1e-6))'
            elif node.op == 'sigmoid':
                return f'(1.0 / (1.0 + np.exp(-np.clip({child_str}, -100, 100))))'
            elif node.op == 'tanh':
                return f'np.tanh({child_str})'
            elif node.op == 'cbrt':
                return f'(np.sign({child_str}) * np.abs({child_str})**(1/3))'
            elif node.op == 'fourth_root':
                return f'np.abs({child_str})**0.25'
            elif node.op == 'floor':
                return f'np.floor({child_str})'
            elif node.op == 'ceil':
                return f'np.ceil({child_str})'
            elif node.op == 'round':
                return f'np.round({child_str})'
            elif node.op == 'arcsin':
                return f'np.arcsin(np.clip({child_str}, -1, 1))'
            elif node.op == 'arccos':
                return f'np.arccos(np.clip({child_str}, -1, 1))'
            elif node.op == 'arctan':
                return f'np.arctan({child_str})'
            elif node.op == 'sinh':
                return f'np.sinh(np.clip({child_str}, -100, 100))'
            elif node.op == 'cosh':
                return f'np.cosh(np.clip({child_str}, -100, 100))'
            elif node.op == 'fifth_root':
                return f'(np.sign({child_str}) * np.abs({child_str})**0.2)'
            elif node.op == 'sixth_root':
                return f'np.abs({child_str})**(1/6)'
            else:
                return f'{node.op}({child_str})'
        
        elif op_obj.arity == 2:
            left = tree_to_lambda_string(node.children[0])
            right = tree_to_lambda_string(node.children[1])
            
            if node.op == 'add':
                return f'({left} + {right})'
            elif node.op == 'sub':
                return f'({left} - {right})'
            elif node.op == 'mul':
                return f'({left} * {right})'
            elif node.op == 'div':
                return f'({left} / (np.abs({right}) + 1e-6))'
            elif node.op == 'pow':
                return f'(np.abs({left})**np.clip({right}, -10, 10))'
            elif node.op == 'mod':
                return f'np.mod({left}, np.abs({right}) + 1e-6)'
            elif node.op == 'min':
                return f'np.minimum({left}, {right})'
            elif node.op == 'max':
                return f'np.maximum({left}, {right})'
            elif node.op == 'avg':
                return f'(({left} + {right}) / 2.0)'
            elif node.op == 'weighted_avg':
                return f'(0.7 * {left} + 0.3 * {right})'
            elif node.op == 'geometric_mean':
                return f'np.sqrt(np.abs({left} * {right}))'
            elif node.op == 'harmonic_mean':
                return f'(2.0 / (1.0/(np.abs({left}) + 1e-6) + 1.0/(np.abs({right}) + 1e-6)))'
            elif node.op == 'hypot':
                return f'np.sqrt({left}**2 + {right}**2)'
            elif node.op == 'atan2':
                return f'np.arctan2({left}, {right})'
            elif node.op == 'gcd_like':
                return f'np.minimum(np.abs({left}) + 0.1, np.abs({right}) + 0.1)'
            elif node.op == 'lcm_like':
                return f'np.maximum(np.abs({left}) + 0.1, np.abs({right}) + 0.1)'
            else:
                return f'{node.op}({left}, {right})'
        
        elif op_obj.arity == 3:
            args = [tree_to_lambda_string(child) for child in node.children]
            
            if node.op == 'if_then_else':
                return f'np.where({args[0]} > 0, {args[1]}, {args[2]})'
            elif node.op == 'clamp':
                return f'np.clip({args[0]}, {args[1]}, {args[2]})'
            elif node.op == 'lerp':
                return f'({args[0]} + np.clip({args[2]}, 0, 1) * ({args[1]} - {args[0]}))'
            elif node.op == 'weighted_sum':
                return f'(0.5 * {args[0]} + 0.3 * {args[1]} + 0.2 * {args[2]})'
            else:
                return f'{node.op}({", ".join(args)})'
    
    return 'np.nan'

def load_equations_from_file(filepath: str) -> List[gae.Equation]:
    """Load equations from saved checkpoint file"""
    
    if not os.path.exists(filepath):
        return []
    
    try:
        # Import the module dynamically
        import importlib.util
        spec = importlib.util.spec_from_file_location("checkpoint", filepath)
        if spec and spec.loader:
            checkpoint = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(checkpoint)
            
            if hasattr(checkpoint, 'EQUATIONS_TO_TEST'):
                print(f"  Found {len(checkpoint.EQUATIONS_TO_TEST)} equations in checkpoint")
                # Convert back to Equation objects (simplified - we'll just use them as initial population)
                # The actual equations will be re-created, but we can track metadata
                return []  # Return empty for now, will use equations_dict directly
        
        return []
    except Exception as e:
        print(f"  Warning: Could not load checkpoint from {filepath}: {e}")
        return []

# ============================================================================
# EQUATION EVALUATION
# ============================================================================

def evaluate_equations(equations_dict: Dict[str, Dict], 
                      all_benchmarks_data: Dict[str, Dict],
                      allowed_vars: List[str] = None,
                      verbose: bool = True) -> Tuple[List[Tuple[str, Dict]], Dict[str, int]]:
    """
    Evaluate all equations and return results sorted by performance.
    
    Args:
        equations_dict: Dictionary of equation_id -> equation_info
        all_benchmarks_data: Dictionary of benchmark_name -> data_dict
        allowed_vars: List of allowed variable names (default: ['complexity', 'count'])
        verbose: Whether to print progress messages
    
    Returns: 
        - List of (equation_id, results_dict) tuples sorted by avg correlation
        - Dictionary with failure reasons and counts
    """
    
    if allowed_vars is None:
        allowed_vars = ['complexity', 'count']
    
    results = []
    
    # Track failure reasons
    failure_stats = {
        'params_nan_inf': 0,
        'params_too_large': 0,
        'predictions_nan_inf': 0,
        'predictions_overflow': 0,
        'predictions_constant': 0,
        'benchmark_constant': 0,
        'correlation_nan_inf': 0,
        'corrcoef_shape_error': 0,
        'curve_fit_exception': 0,
        'other_exception': 0
    }
    
    # Get all benchmark data for all allowed variables
    all_benchmark_values = np.concatenate([data['benchmark_values'] for data in all_benchmarks_data.values()])
    
    # Dynamically extract variable data
    all_var_values = {}
    for var in allowed_vars:
        var_data = []
        for data in all_benchmarks_data.values():
            if var in data:
                var_data.append(data[var])
            else:
                # If variable not in data, use zeros as placeholder
                var_data.append(np.zeros_like(data['benchmark_values']))
        all_var_values[var] = np.concatenate(var_data)
    
    total_eqs = len(equations_dict)
    
    for eq_idx, (eq_id, eq_info) in enumerate(equations_dict.items()):
        if verbose and eq_idx % max(1, total_eqs // 10) == 0:
            print(f"  Evaluating equations... {eq_idx}/{total_eqs} ({100*eq_idx/total_eqs:.1f}%)")
        
        try:
            # Fit equation parameters once across ALL benchmarks (shared parameters)
            def fit_func(x, *params):
                # x is a matrix where each row is a variable's values
                # Convert to individual variable arguments
                if len(allowed_vars) == 1:
                    # Single variable case - x is 1D
                    var_args = [x]
                else:
                    # Multiple variables - x is 2D, extract each row
                    var_args = [x[i] for i in range(len(allowed_vars))]
                
                result = eq_info['func'](*var_args, *params)
                # Clip extreme values to prevent overflow in curve_fit
                return np.clip(result, -1e100, 1e100)
            
            # Prepare combined data from all benchmarks
            # Stack variable data as rows
            if len(allowed_vars) == 1:
                # For single variable, pass as 1D array
                x_data = all_var_values[allowed_vars[0]]
            else:
                # For multiple variables, stack as 2D array
                x_data = np.vstack([all_var_values[var] for var in allowed_vars])
            
            # Fit the equation parameters to all benchmark data
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                shared_params, _ = curve_fit(
                    fit_func,
                    x_data,
                    all_benchmark_values,
                    p0=eq_info['initial_guess'],
                    maxfev=500000,
                )
            
            # Validate fitted parameters aren't extreme
            if np.any(np.isnan(shared_params)) or np.any(np.isinf(shared_params)):
                failure_stats['params_nan_inf'] += 1
                continue
            if np.any(np.abs(shared_params) > 1e50):
                failure_stats['params_too_large'] += 1
                continue
            
            # Calculate correlation and R² for EACH benchmark separately
            # using the same shared parameters
            benchmark_correlations = []
            all_valid = True
            
            for bench_name, data in all_benchmarks_data.items():
                # Extract variable values for this benchmark
                var_args = [data[var] for var in allowed_vars if var in data]
                
                # Use shared parameters for this benchmark
                predictions = eq_info['func'](*var_args, *shared_params)
                
                # Validate predictions - check for extreme values first
                if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                    failure_stats['predictions_nan_inf'] += 1
                    all_valid = False
                    break
                
                # Check for overflow (very large values)
                if np.any(np.abs(predictions) > 1e100):
                    failure_stats['predictions_overflow'] += 1
                    all_valid = False
                    break
                
                # Check for constant predictions
                pred_std = np.std(predictions)
                if pred_std < 1e-10 or np.isnan(pred_std):
                    failure_stats['predictions_constant'] += 1
                    all_valid = False
                    break
                
                # Check benchmark values std too (for corrcoef)
                bench_std = np.std(data['benchmark_values'])
                if bench_std < 1e-10 or np.isnan(bench_std):
                    failure_stats['benchmark_constant'] += 1
                    all_valid = False
                    break
                
                # Calculate correlation with error handling
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    corr_matrix = np.corrcoef(data['benchmark_values'], predictions)
                    if corr_matrix.shape != (2, 2):
                        failure_stats['corrcoef_shape_error'] += 1
                        all_valid = False
                        break
                    correlation = corr_matrix[0, 1]
                
                if np.isnan(correlation) or np.isinf(correlation):
                    failure_stats['correlation_nan_inf'] += 1
                    all_valid = False
                    break
                
                # Use raw correlation (not absolute value) - negative = bad
                benchmark_correlations.append(correlation)
            
            if not all_valid:
                continue
            
            # Calculate average correlation
            avg_correlation = np.mean(benchmark_correlations)
            
            results.append((eq_id, {
                'avg_correlation': avg_correlation,
                'params': shared_params.tolist(),
                'name': eq_info['name'],
                'initial_guess': eq_info['initial_guess'],
                'func': eq_info['func'],
                'benchmark_correlations': benchmark_correlations
            }))
            
        except Exception as e:
            # Track different exception types
            exception_type = type(e).__name__
            if 'RuntimeError' in exception_type or 'OptimizeWarning' in exception_type:
                failure_stats['curve_fit_exception'] += 1
            elif 'UFuncTypeError' in exception_type or 'TypeError' in exception_type:
                failure_stats['other_exception'] += 1
            else:
                failure_stats['other_exception'] += 1
            
            # Only print first few errors with details for debugging
            total_errors = sum(failure_stats.values())
            if verbose and total_errors <= 3:
                print(f"    Skipped {eq_id}: {exception_type}")
                print(f"      Equation: {eq_info.get('name', 'unknown')[:100]}")
                print(f"      Error: {str(e)[:150]}")
            elif verbose and total_errors % 50 == 0:
                # Print summary every 50 errors
                print(f"    ... {total_errors} equations skipped so far ...")
            continue
    
    # Sort by average correlation (descending)
    results.sort(key=lambda x: x[1]['avg_correlation'], reverse=True)
    
    if verbose:
        print(f"  Successfully evaluated {len(results)}/{total_eqs} equations")
    
    return results, failure_stats

# ============================================================================
# EVOLUTION LOOP
# ============================================================================

def run_evolution(appended_benchmarks_df: pd.DataFrame,
                 bench_rows_names: List[str],
                 output_folder: str,
                 min_stopping_correlation: float = 0.6,
                 population_size: int = 50,
                 max_generations: int = 100,
                 top_n_to_keep: int = 10,
                 elite_size: int = 5,
                 mandatory_vars: List[str] = None,
                 allowed_vars: List[str] = None,
                 simplicity_weight: float = 0.1,
                 resume_from_checkpoint: str = None,
                 max_stagnation: int = 15,
                 adaptive_mutation: bool = True,
                 diversity_injection_rate: float = 0.2):
    """
    Main evolution loop:
    1. Load/generate initial equations (optionally from checkpoint)
    2. Evaluate them
    3. Select top performers
    4. Use GA to create variations
    5. Repeat until stopping criteria
    
    Args:
        mandatory_vars: List of variable names that MUST appear in every equation (e.g., ['complexity'])
        allowed_vars: List of variable names that CAN be used in equations (restricts available variables)
        simplicity_weight: Weight for simplicity in multi-objective optimization (higher = prefer simpler)
        resume_from_checkpoint: Path to checkpoint file to resume from (e.g., 'top_equations_gen10.py')
        max_stagnation: Maximum generations without improvement before stopping (default: 15)
        adaptive_mutation: Increase mutation rate when stagnating (default: True)
        diversity_injection_rate: Fraction of population to replace with random when stagnating (default: 0.2)
    """
    
    print("\n" + "="*80)
    print("STARTING EVOLUTIONARY EQUATION DISCOVERY")
    print("="*80)
    print(f"Population size: {population_size}")
    print(f"Max generations: {max_generations}")
    print(f"Target correlation: {min_stopping_correlation}")
    print(f"Elite size: {elite_size}")
    print(f"Top equations to keep: {top_n_to_keep}")
    print(f"Mandatory variables: {mandatory_vars or 'None'}")
    print(f"Allowed variables: {allowed_vars or 'All'}")
    print(f"Simplicity weight: {simplicity_weight}")
    print(f"Resume from checkpoint: {resume_from_checkpoint or 'No (fresh start)'}")
    print(f"Max stagnation: {max_stagnation} generations")
    print(f"Adaptive mutation: {'Enabled' if adaptive_mutation else 'Disabled'}")
    print(f"Diversity injection: {diversity_injection_rate*100:.0f}% when stagnating")
    
    # Create output folder
    evolution_folder = os.path.join(output_folder, 'equation_evolution')
    os.makedirs(evolution_folder, exist_ok=True)
    
    # Determine which variables to use (default to complexity and count if not specified)
    if allowed_vars is None:
        allowed_vars = ['complexity', 'count']
    
    # Prepare benchmark data
    print("\nPreparing benchmark data...")
    all_benchmarks_data = {}
    for bench_name in bench_rows_names:
        # Build mask for all required variables
        mask_conditions = [appended_benchmarks_df[bench_name].isna()]
        for var in allowed_vars:
            if var in appended_benchmarks_df.columns:
                mask_conditions.append(appended_benchmarks_df[var].isna())
        
        mask_bench = ~np.logical_or.reduce(mask_conditions)
        
        # Extract data for all allowed variables
        benchmark_values = appended_benchmarks_df.loc[mask_bench, bench_name].values
        var_data = {}
        for var in allowed_vars:
            if var in appended_benchmarks_df.columns:
                var_data[var] = appended_benchmarks_df.loc[mask_bench, var].values
        
        if len(benchmark_values) >= 10:
            all_benchmarks_data[bench_name] = {
                'benchmark_values': benchmark_values,
                **var_data  # Unpack all variable data
            }
            print(f"  {bench_name}: {len(benchmark_values)} data points")
    
    print(f"\nTotal benchmarks ready: {len(all_benchmarks_data)}")
    
    # Try to find the latest checkpoint if resume requested
    checkpoint_equations = None
    if resume_from_checkpoint:
        # Handle 'auto' keyword to automatically find latest checkpoint
        if resume_from_checkpoint.lower() == 'auto':
            print(f"\nSearching for latest checkpoint in {evolution_folder}...")
            checkpoint_path = None
        else:
            checkpoint_path = os.path.join(evolution_folder, resume_from_checkpoint)
        
        # Search for latest checkpoint if path not specified or doesn't exist
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            if checkpoint_path and checkpoint_path != os.path.join(evolution_folder, 'auto'):
                print(f"\nCheckpoint '{resume_from_checkpoint}' not found at: {checkpoint_path}")
                print(f"Searching for latest checkpoint in {evolution_folder}...")
            
            # Look for checkpoint files
            if not os.path.exists(evolution_folder):
                print(f"  Evolution folder doesn't exist yet: {evolution_folder}")
                print(f"  Starting fresh without checkpoint.")
                checkpoint_path = None
            else:
                checkpoint_files = [f for f in os.listdir(evolution_folder) 
                                   if f.startswith('top_equations_gen') and f.endswith('.py')]
                if checkpoint_files:
                    # Extract generation numbers and find the latest
                    gen_numbers = []
                    for f in checkpoint_files:
                        try:
                            gen_num = int(f.replace('top_equations_gen', '').replace('.py', ''))
                            gen_numbers.append((gen_num, f))
                        except:
                            continue
                    if gen_numbers:
                        latest_gen, latest_file = max(gen_numbers, key=lambda x: x[0])
                        checkpoint_path = os.path.join(evolution_folder, latest_file)
                        print(f"  ✅ Found checkpoint: {latest_file} (generation {latest_gen})")
                    else:
                        print(f"  No valid checkpoint files found.")
                        checkpoint_path = None
                else:
                    print(f"  No checkpoint files found in {evolution_folder}")
                    checkpoint_path = None
        
        # Load checkpoint if path was found
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"\nLoading checkpoint from: {checkpoint_path}")
            try:
                import importlib.util
                import json
                spec = importlib.util.spec_from_file_location("checkpoint", checkpoint_path)
                if spec and spec.loader:
                    checkpoint = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(checkpoint)
                    
                    if hasattr(checkpoint, 'TREE_STRUCTURES') and hasattr(checkpoint, 'EQUATION_METADATA'):
                        # New format with tree structures - properly reconstruct equations
                        print(f"  Found {len(checkpoint.TREE_STRUCTURES)} equations with tree structures")
                        checkpoint_equations = []
                        for tree_dict, metadata in zip(checkpoint.TREE_STRUCTURES, checkpoint.EQUATION_METADATA):
                            # Reconstruct tree
                            tree = gae.ExprNode.from_dict(tree_dict)
                            # Reconstruct equation
                            eq = gae.Equation(
                                tree=tree,
                                num_params=metadata['num_params'],
                                param_initial_guess=metadata['param_initial_guess'],
                                fitness=metadata['fitness'],
                                avg_correlation=metadata['avg_correlation'],
                                avg_r2=metadata['avg_r2'],
                                simplicity_score=metadata['simplicity_score'],
                                generation=metadata['generation']
                            )
                            checkpoint_equations.append(eq)
                        print(f"  Successfully reconstructed {len(checkpoint_equations)} equations")
                        print(f"  Best checkpoint correlation: {max(eq.avg_correlation for eq in checkpoint_equations):.6f}")
                    elif hasattr(checkpoint, 'EQUATIONS_TO_TEST'):
                        # Old format without tree structures - can't properly resume
                        print(f"  WARNING: Old checkpoint format detected (no tree structures)")
                        print(f"  Cannot resume evolution - starting fresh")
                        print(f"  (To enable resume, re-run and save new checkpoints)")
                        checkpoint_equations = None
            except Exception as e:
                import traceback
                print(f"  ⚠️  ERROR: Could not load checkpoint: {type(e).__name__}: {e}")
                print(f"  Details: {traceback.format_exc().splitlines()[-3]}")
                print(f"  This may be due to:")
                print(f"    - Old checkpoint format with 'Infinity' values")
                print(f"    - Corrupted checkpoint file")
                print(f"    - Incompatible Python/numpy versions")
                print(f"  Starting fresh...")
                checkpoint_equations = None
    
    # Initialize population
    print("\nInitializing population...")
    population = gae.Population(size=population_size, max_depth=4, num_params=5, 
                               mandatory_vars=mandatory_vars, allowed_vars=allowed_vars, 
                               simplicity_weight=simplicity_weight)
    
    # If we have checkpoint equations, use them to seed the population
    if checkpoint_equations:
        print(f"Seeding population with {len(checkpoint_equations)} checkpoint equations...")
        # Use checkpoint equations directly
        for eq in checkpoint_equations[:population_size]:
            population.equations.append(eq.copy())
        
        # Fill remaining slots if needed
        while len(population.equations) < population_size:
            eq = gae.generate_random_equation(max_depth=4, num_params=5, 
                                             mandatory_vars=mandatory_vars, 
                                             allowed_vars=allowed_vars)
            population.equations.append(eq)
        
        print(f"  Population seeded: {len(checkpoint_equations)} from checkpoint + {population_size - len(checkpoint_equations)} new random")
        
        # Update generation number to continue from checkpoint
        max_gen = max(eq.generation for eq in checkpoint_equations)
        population.generation = max_gen + 1
        print(f"  Continuing from generation {population.generation}")
    else:
        # No checkpoint - initialize fresh
        population.initialize(None)
        print(f"Generated {len(population.equations)} initial equations")
    
    # Evolution loop
    best_correlation_ever = 0.0
    generations_without_improvement = 0
    base_mutation_rate = 0.3
    current_mutation_rate = base_mutation_rate
    
    # If resuming from checkpoint, initialize best_correlation_ever
    if checkpoint_equations:
        best_correlation_ever = max(eq.avg_correlation for eq in checkpoint_equations)
        print(f"  Starting with best correlation from checkpoint: {best_correlation_ever:.6f}")
    
    history = []
    
    # Start generation from checkpoint if resuming
    start_generation = population.generation if checkpoint_equations else 0
    
    for generation in range(start_generation, max_generations):
        gen_start_time = time.time()
        
        print(f"\n{'='*80}")
        print(f"GENERATION {generation}")
        print(f"{'='*80}")

        # Convert equations to dict format
        equations_dict, conversion_failures = equations_to_dict(population.equations, 
                                                                 allowed_vars=allowed_vars, 
                                                                 debug=False)
        
        # Total failures = equations that didn't make it to dict
        total_conversion_failures = len(population.equations) - len(equations_dict)
        tracked_conversion_failures = sum(conversion_failures.values())
        
        # Filter out invalid equations before evaluation
        # Use the equation_object directly from the dict (already stored there)
        valid_equations_dict = {}
        filtered_stats = {
            'degenerate': 0,
            'no_variables': 0,
            'duplicate': 0,
            'other_invalid': 0
        }
        seen_forms = set()
        
        for eq_id, eq_info in equations_dict.items():
            # Get equation object directly from dict (stored in equations_to_dict)
            eq_obj = eq_info.get('equation_object')
            if eq_obj:
                # Check for duplicates
                canonical = eq_obj.get_canonical_form()
                if canonical in seen_forms:
                    filtered_stats['duplicate'] += 1
                    continue
                
                # Check specific invalidity reasons
                if eq_obj.is_degenerate():
                    filtered_stats['degenerate'] += 1
                elif len(eq_obj.get_all_variables()) == 0:
                    filtered_stats['no_variables'] += 1
                elif not eq_obj.is_valid():
                    filtered_stats['other_invalid'] += 1
                else:
                    valid_equations_dict[eq_id] = eq_info
                    seen_forms.add(canonical)
            else:
                # No equation object, include anyway (shouldn't happen)
                valid_equations_dict[eq_id] = eq_info
        
        total_filtered = sum(filtered_stats.values())
        
        if len(valid_equations_dict) == 0:
            print("⚠️  All equations are invalid! Reinitializing population...")
            population.initialize()
            continue
        
        # Evaluate all valid equations
        print("Evaluating equations...")
        eval_results, failure_stats = evaluate_equations(valid_equations_dict, all_benchmarks_data, 
                                                          allowed_vars=allowed_vars, verbose=True)
        
        if not eval_results:
            print("⚠️  No valid equations in this generation! Reinitializing...")
            population.initialize()
            generations_without_improvement += 1
            continue
        
        # First, reset ALL fitness values to ensure no stale data
        # This prevents equations that fail current evaluation from keeping old fitness scores
        successfully_evaluated_ids = {valid_equations_dict[eq_id]['equation_object'].unique_id 
                                       for eq_id, _ in eval_results 
                                       if eq_id in valid_equations_dict and valid_equations_dict[eq_id].get('equation_object')}
        
        for eq in population.equations:
            if eq.unique_id not in successfully_evaluated_ids:
                # Reset fitness for equations not successfully evaluated this generation
                eq.fitness = -1.0
                eq.avg_correlation = 0.0
        
        # Update fitness scores for successfully evaluated equations
        # Fitness = correlation - simplicity_penalty (multi-objective optimization)
        valid_eqs_updated = 0
        eval_result_ids = {eq_id for eq_id, _ in eval_results}
        
        for eq_id, res in eval_results:
            # Find the equation object from the dict
            if eq_id in valid_equations_dict:
                eq_obj = valid_equations_dict[eq_id].get('equation_object')
                if eq_obj:
                    # Store correlation separately
                    eq_obj.avg_correlation = res['avg_correlation']
                    # Calculate fitness as: correlation - simplicity_weight * simplicity
                    # This balances accuracy with simplicity
                    eq_obj.fitness = (res['avg_correlation'] ** 2) - ((simplicity_weight * (1 + eq_obj.simplicity_score)) ** 2)
                    valid_eqs_updated += 1
        
        # Track equations that made it to evaluation but didn't return results
        # These failed during evaluation but their failure reason wasn't captured in failure_stats
        sent_to_eval = len(valid_equations_dict)
        returned_from_eval = len(eval_results)
        eval_silent_failures = sent_to_eval - returned_from_eval
        
        print(f"  Updated fitness for {valid_eqs_updated}/{len(population.equations)} equations")
        if valid_eqs_updated < len(population.equations):
            not_updated = len(population.equations) - valid_eqs_updated
            print(f"  ({not_updated} equations marked invalid or failed evaluation, fitness set to -1.0)")
            
            # Print detailed breakdown of failures during evaluation
            total_eval_failures = sum(failure_stats.values())
            total_pre_filtered = sum(filtered_stats.values())
            # Use the ACTUAL conversion failures (already calculated earlier), not just tracked ones
            actual_conversion_failed = total_conversion_failures
            
            # Note: eval_silent_failures represents equations that reached evaluate_equations()
            # but didn't return in eval_results AND weren't counted in failure_stats
            # This can happen if evaluate_equations has early returns or uncaught exceptions
            
            if total_eval_failures > 0 or total_pre_filtered > 0 or actual_conversion_failed > 0 or eval_silent_failures > 0:
                print(f"  Failure breakdown:")
                
                # Show conversion failures (equations that failed to convert to dict)
                if actual_conversion_failed > 0:
                    print(f"    Conversion failures ({actual_conversion_failed}):")
                    if tracked_conversion_failures > 0:
                        print(f"      • Tracked: {tracked_conversion_failures}")
                        for reason, count in sorted(conversion_failures.items(), key=lambda x: x[1], reverse=True):
                            if count > 0:
                                readable_reason = reason.replace('_', ' ').title()
                                print(f"        - {readable_reason}: {count}")
                    untracked_conv = actual_conversion_failed - tracked_conversion_failures
                    if untracked_conv > 0:
                        print(f"      • Untracked: {untracked_conv}")
                
                # Show pre-filtering reasons
                if total_pre_filtered > 0:
                    print(f"    Pre-filtered ({total_pre_filtered}):")
                    for reason, count in sorted(filtered_stats.items(), key=lambda x: x[1], reverse=True):
                        if count > 0:
                            readable_reason = reason.replace('_', ' ').title()
                            print(f"      • {readable_reason}: {count}")
                
                # Show equations that failed during evaluation
                # Note: eval_silent_failures includes ALL equations that didn't return from eval
                # Some of these were tracked in failure_stats, some might be truly silent
                if eval_silent_failures > 0:
                    print(f"    Failed during evaluation ({eval_silent_failures}):")
                    if total_eval_failures > 0:
                        print(f"      • Tracked failures: {total_eval_failures}")
                        for reason, count in sorted(failure_stats.items(), key=lambda x: x[1], reverse=True):
                            if count > 0:
                                readable_reason = reason.replace('_', ' ').title()
                                print(f"        - {readable_reason}: {count}")
                    
                    untracked = eval_silent_failures - total_eval_failures
                    if untracked > 0:
                        print(f"      • Untracked failures: {untracked}")
                        print(f"        (These failed but weren't counted in failure_stats)")
                
                # Sanity check: all categories should sum to not_updated
                # Note: eval_silent_failures already includes total_eval_failures, so don't double-count
                expected_total = actual_conversion_failed + total_pre_filtered + eval_silent_failures
                if expected_total != not_updated:
                    # Account for rounding/edge cases with tolerance of 1
                    diff = abs(expected_total - not_updated)
                    if diff > 1:
                        print(f"    ⚠️  Accounting mismatch: {diff} equations unaccounted")
                        print(f"        (Expected {not_updated}, got {expected_total})")
                        print(f"        Breakdown: conv={actual_conversion_failed}, pre={total_pre_filtered}, silent={eval_silent_failures}, tracked={total_eval_failures}")
        
        # Get statistics
        stats = population.get_statistics()
        
        # Print top 5 equations FROM CURRENT EVALUATION
        # print(f"\nTop 5 equations (from current evaluation):")
        # for i, (eq_id, res) in enumerate(eval_results[:5], 1):
        #     # Get the equation object from the valid_equations_dict
        #     eq_obj = valid_equations_dict.get(eq_id, {}).get('equation_object')
        #     if eq_obj:
        #         simplicity = eq_obj.simplicity_score
        #         combined_fitness = eq_obj.fitness
        #         multi_obj = combined_fitness - simplicity_weight * simplicity
                
        #         print(f"  {i}. Fitness: {combined_fitness:.6f}, Corr: {res['avg_correlation']:.6f}, Simplicity: {simplicity:.2f}")
        #         print(f"     Multi-obj: {multi_obj:.6f} | {res['name'][:80]}")
        #     else:
        #         print(f"  {i}. Corr: {res['avg_correlation']:.6f}")
        #         print(f"     {res['name'][:80]}")
        
        # Print top 5 equations FROM ENTIRE POPULATION (including previous generations)
        print(f"\nTop 5 equations (from entire population):")
        population_sorted = sorted(population.equations, 
                                   key=lambda eq: eq.fitness if eq.fitness > -np.inf else -1.0, 
                                   reverse=True)
        for i, eq in enumerate(population_sorted[:5], 1):
            if eq.fitness > -np.inf:
                simplicity = eq.simplicity_score
                # Fitness already includes simplicity penalty
                print(f"  {i}. Fitness: {eq.fitness:.6f}, Corr: {eq.avg_correlation:.6f}, Simplicity: {simplicity:.2f}")
                print(f"     | {eq.to_string()[:80]}")
            else:
                print(f"  {i}. [Invalid fitness]")
        
        print(f"\nGeneration statistics:")
        print(f"  Best fitness: {stats['best_fitness']:.6f}")
        print(f"  Best correlation: {stats['best_correlation']:.6f}")
        print(f"  Avg fitness: {stats['avg_fitness']:.6f}")
        print(f"  Avg complexity: {stats['avg_complexity']:.2f}")
        print(f"  Avg tree depth: {stats['avg_depth']:.1f}")
        print(f"  Diversity: {stats['unique_equations']}/{stats['size']} unique ({stats['diversity']*100:.1f}%)")
        
        # Track best ever using combined fitness (not just correlation)
        if stats['best_fitness'] > best_correlation_ever:
            best_correlation_ever = stats['best_fitness']
            generations_without_improvement = 0
            current_mutation_rate = base_mutation_rate  # Reset mutation rate on improvement
            print(f"  🎉 NEW BEST FITNESS: {best_correlation_ever:.6f}")
        else:
            generations_without_improvement += 1
            
            # Adaptive mutation: increase when stagnating
            if adaptive_mutation and generations_without_improvement > 3:
                current_mutation_rate = min(0.6, base_mutation_rate + 0.05 * (generations_without_improvement - 3))
                print(f"  ⚠️  No improvement for {generations_without_improvement} generations (mutation rate: {current_mutation_rate:.2f})")
            else:
                print(f"  No improvement for {generations_without_improvement} generations")
        
        # Diversity injection: add fresh random equations when stagnating
        if generations_without_improvement > 0 and generations_without_improvement % 5 == 0:
            n_inject = int(population_size * diversity_injection_rate)
            print(f"  💉 DIVERSITY INJECTION: Adding {n_inject} fresh random equations")
            
            # Replace worst equations with new random ones
            # Sort by fitness (which already includes simplicity penalty)
            population.equations.sort(key=lambda eq: eq.fitness if eq.fitness > -np.inf else -1.0, reverse=True)
            for i in range(n_inject):
                idx = -(i+1)  # Replace from worst
                new_eq = gae.generate_random_equation(4, 5, mandatory_vars, allowed_vars)
                new_eq.generation = generation
                population.equations[idx] = new_eq
        
        # Save history
        history.append({
            'generation': generation,
            'best_correlation': stats['best_correlation'],
            'avg_correlation': stats['avg_fitness'],
            'avg_complexity': stats['avg_complexity'],
            'time': time.time() - gen_start_time
        })
        
        # Check stopping criteria
        if best_correlation_ever >= min_stopping_correlation:
            print(f"\n🎉 TARGET CORRELATION REACHED: {best_correlation_ever:.6f} >= {min_stopping_correlation}")
            break
        
        if generations_without_improvement >= max_stagnation:
            print(f"\n⚠️  Evolution stagnated for {max_stagnation} generations.")
            print(f"   Best correlation achieved: {best_correlation_ever:.6f}")
            print(f"   Consider:")
            print(f"   - Increasing population_size (current: {population_size})")
            print(f"   - Reducing simplicity_weight (current: {simplicity_weight})")
            print(f"   - Increasing max_stagnation (current: {max_stagnation})")
            print(f"   Stopping evolution.")
            break
        
        # Save top equations to file every 10 generations
        if generation % 10 == 0 or generation == max_generations - 1:
            print(f"\nSaving top {top_n_to_keep} equations to file...")
            top_equations = [population.equations[i] for i in range(min(top_n_to_keep, len(population.equations)))]
            save_file = os.path.join(evolution_folder, f'top_equations_gen{generation}.py')
            save_equations_to_file(top_equations, save_file)
            print(f"  Saved to: {save_file}")
        
        # Evolve to next generation
        print("\nEvolving to next generation...")
        population.evolve(elite_size=elite_size, crossover_rate=0.7, mutation_rate=current_mutation_rate, stagnation_counter=generations_without_improvement)
        
        gen_time = time.time() - gen_start_time
        print(f"\nGeneration {generation} completed in {gen_time:.2f}s")
        print(f"Current mutation rate: {current_mutation_rate:.2f}")
    
    # Final results
    print(f"\n{'='*80}")
    print("EVOLUTION COMPLETE")
    print(f"{'='*80}")
    print(f"Total generations: {len(history)}")
    print(f"Best correlation achieved: {best_correlation_ever:.6f}")
    print(f"Target correlation: {min_stopping_correlation}")
    
    # Save final best equations
    print(f"\nSaving final top {top_n_to_keep} equations...")
    population.equations.sort(key=lambda eq: eq.fitness, reverse=True)
    top_equations = population.equations[:top_n_to_keep]
    
    final_file = os.path.join(evolution_folder, 'z_eqs_to_test.py')
    save_equations_to_file(top_equations, final_file)
    print(f"Saved to: {final_file}")
    
    # Save evolution history
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(evolution_folder, 'evolution_history.csv'), index=False)
    
    # Print final top equations
    print(f"\nFinal Top {min(5, len(top_equations))} Equations:")
    for i, eq in enumerate(top_equations[:5], 1):
        variables_used = eq.get_all_variables()
        mandatory_check = "✓" if all(var in variables_used for var in (mandatory_vars or [])) else "✗"
        
        print(f"\n{i}. Correlation: {eq.avg_correlation:.6f}, Simplicity: {eq.simplicity_score:.2f}")
        print(f"   Multi-objective score: {eq.avg_correlation - simplicity_weight * eq.simplicity_score:.6f}")
        print(f"   Complexity: {eq.get_complexity():.2f}, Depth: {eq.get_depth()}, Size: {eq.get_size()}")
        print(f"   Variables: {variables_used}, Mandatory: {mandatory_check}")
        print(f"   {eq.to_string()[:100]}")
    
    return top_equations, history

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("This module should be imported and used with d_getKnowledge.py")
    gbb.print_building_blocks_summary()
