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

def equations_to_dict(equations: List[gae.Equation]) -> Dict[str, Dict]:
    """Convert list of Equation objects to dictionary format for z_eqs_to_test.py"""
    
    result = {}
    
    for idx, eq in enumerate(equations):
        eq_id = f"ga_gen{eq.generation}_eq{idx}"
        
        # Create the lambda function
        func = eq.to_lambda()
        
        # Create human-readable name
        name = eq.to_string()
        
        # Get initial guess
        initial_guess = eq.param_initial_guess
        
        result[eq_id] = {
            'func': func,
            'name': name,
            'initial_guess': initial_guess,
            'metadata': {
                'generation': eq.generation,
                'fitness': float(eq.fitness),
                'avg_correlation': float(eq.avg_correlation),
                'avg_r2': float(eq.avg_r2),
                'complexity': float(eq.get_complexity()),
                'depth': eq.get_depth(),
                'size': eq.get_size(),
                'parent_ids': eq.parent_ids
            }
        }
    
    return result

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
            metadata = {
                'num_params': eq.num_params,
                'param_initial_guess': eq.param_initial_guess,
                'fitness': eq.fitness,
                'avg_correlation': eq.avg_correlation,
                'avg_r2': eq.avg_r2,
                'simplicity_score': eq.simplicity_score,
                'generation': eq.generation
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
                      verbose: bool = True) -> List[Tuple[str, Dict]]:
    """
    Evaluate all equations and return results sorted by performance.
    
    Returns: List of (equation_id, results_dict) tuples sorted by avg correlation
    """
    
    results = []
    
    # Get all benchmark data
    all_benchmark_values = np.concatenate([data['benchmark_values'] for data in all_benchmarks_data.values()])
    all_complexity_values = np.concatenate([data['complexity_values'] for data in all_benchmarks_data.values()])
    all_count_values = np.concatenate([data['count_values'] for data in all_benchmarks_data.values()])
    
    total_eqs = len(equations_dict)
    
    for eq_idx, (eq_id, eq_info) in enumerate(equations_dict.items()):
        if verbose and eq_idx % max(1, total_eqs // 10) == 0:
            print(f"  Evaluating equations... {eq_idx}/{total_eqs} ({100*eq_idx/total_eqs:.1f}%)")
        
        try:
            # Create wrapper function for curve_fit
            def fit_func(x, *params):
                complexity = x[0]
                count = x[1]
                return eq_info['func'](complexity, count, *params)
            
            # Prepare data for curve_fit
            x_data = np.vstack([all_complexity_values, all_count_values])
            
            # Fit the equation with all data
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                shared_params, _ = curve_fit(
                    fit_func,
                    x_data,
                    all_benchmark_values,
                    p0=eq_info['initial_guess'],
                    maxfev=5000000
                )
            
            # Calculate correlation for each benchmark
            benchmark_correlations = []
            benchmark_r2_scores = []
            all_valid = True
            
            for bench_name, data in all_benchmarks_data.items():
                predictions = eq_info['func'](data['complexity_values'], data['count_values'], *shared_params)
                
                # Validate predictions
                if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                    all_valid = False
                    break
                
                if np.std(predictions) < 1e-10:
                    all_valid = False
                    break
                
                # Calculate metrics
                ss_res = np.sum((data['benchmark_values'] - predictions)**2)
                ss_tot = np.sum((data['benchmark_values'] - np.mean(data['benchmark_values']))**2)
                r2 = 1 - (ss_res / ss_tot)
                
                correlation = np.corrcoef(data['benchmark_values'], predictions)[0, 1]
                
                if np.isnan(correlation) or np.isinf(correlation):
                    all_valid = False
                    break
                
                benchmark_correlations.append(abs(correlation))
                benchmark_r2_scores.append(r2)
            
            if not all_valid:
                continue
            
            # Calculate averages
            avg_correlation = np.mean(benchmark_correlations)
            avg_r2 = np.mean(benchmark_r2_scores)
            
            results.append((eq_id, {
                'avg_correlation': avg_correlation,
                'avg_r2': avg_r2,
                'params': shared_params.tolist(),
                'name': eq_info['name'],
                'initial_guess': eq_info['initial_guess'],
                'func': eq_info['func'],
                'benchmark_correlations': benchmark_correlations,
                'benchmark_r2_scores': benchmark_r2_scores
            }))
            
        except Exception as e:
            # Skip failed equations
            if verbose:
                print(f"    Skipped {eq_id}: {type(e).__name__}")
            continue
    
    # Sort by average correlation (descending)
    results.sort(key=lambda x: x[1]['avg_correlation'], reverse=True)
    
    if verbose:
        print(f"  Successfully evaluated {len(results)}/{total_eqs} equations")
    
    return results

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
    print(f"Simplicity weight: {simplicity_weight}")
    print(f"Resume from checkpoint: {resume_from_checkpoint or 'No (fresh start)'}")
    print(f"Max stagnation: {max_stagnation} generations")
    print(f"Adaptive mutation: {'Enabled' if adaptive_mutation else 'Disabled'}")
    print(f"Diversity injection: {diversity_injection_rate*100:.0f}% when stagnating")
    
    # Create output folder
    evolution_folder = os.path.join(output_folder, 'equation_evolution')
    os.makedirs(evolution_folder, exist_ok=True)
    
    # Prepare benchmark data
    print("\nPreparing benchmark data...")
    all_benchmarks_data = {}
    for bench_name in bench_rows_names:
        mask_bench = ~(
            appended_benchmarks_df[bench_name].isna() |
            appended_benchmarks_df['complexity'].isna() |
            appended_benchmarks_df['count'].isna()
        )
        
        benchmark_values = appended_benchmarks_df.loc[mask_bench, bench_name].values
        complexity_values = appended_benchmarks_df.loc[mask_bench, 'complexity'].values
        count_values = appended_benchmarks_df.loc[mask_bench, 'count'].values
        
        if len(benchmark_values) >= 10:
            all_benchmarks_data[bench_name] = {
                'benchmark_values': benchmark_values,
                'complexity_values': complexity_values,
                'count_values': count_values
            }
            print(f"  {bench_name}: {len(benchmark_values)} data points")
    
    print(f"\nTotal benchmarks ready: {len(all_benchmarks_data)}")
    
    # Try to find the latest checkpoint if resume requested
    checkpoint_equations = None
    if resume_from_checkpoint:
        checkpoint_path = os.path.join(evolution_folder, resume_from_checkpoint)
        if not os.path.exists(checkpoint_path):
            # Try to find the latest checkpoint automatically
            print(f"\nCheckpoint '{resume_from_checkpoint}' not found. Searching for latest checkpoint...")
            checkpoint_files = [f for f in os.listdir(evolution_folder) if f.startswith('top_equations_gen') and f.endswith('.py')]
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
                    print(f"  Found checkpoint: {latest_file} (generation {latest_gen})")
        
        # Load checkpoint
        if os.path.exists(checkpoint_path):
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
                print(f"  Warning: Could not load checkpoint: {e}")
                print(f"  Starting fresh...")
                checkpoint_equations = None
    
    # Initialize population
    print("\nInitializing population...")
    population = gae.Population(size=population_size, max_depth=4, num_params=5, 
                               mandatory_vars=mandatory_vars, simplicity_weight=simplicity_weight)
    
    # If we have checkpoint equations, use them to seed the population
    if checkpoint_equations:
        print(f"Seeding population with {len(checkpoint_equations)} checkpoint equations...")
        # Use checkpoint equations directly
        for eq in checkpoint_equations[:population_size]:
            population.equations.append(eq.copy())
        
        # Fill remaining slots if needed
        while len(population.equations) < population_size:
            eq = gae.generate_random_equation(max_depth=4, num_params=5, mandatory_vars=mandatory_vars)
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
    
    print(f"Population size: {len(population.equations)}")
    
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
        equations_dict = equations_to_dict(population.equations)
        
        # Filter out invalid equations before evaluation
        valid_equations_dict = {}
        invalid_count = 0
        for eq_id, eq_info in equations_dict.items():
            # Try to find the corresponding equation object to check validity
            try:
                idx = int(eq_id.split('_eq')[1])
                if idx < len(population.equations):
                    eq_obj = population.equations[idx]
                    if eq_obj.is_valid():
                        valid_equations_dict[eq_id] = eq_info
                    else:
                        invalid_count += 1
            except:
                # If we can't parse, include it anyway
                valid_equations_dict[eq_id] = eq_info
        
        if invalid_count > 0:
            print(f"  Filtered out {invalid_count} invalid/degenerate equations")
        
        if len(valid_equations_dict) == 0:
            print("⚠️  All equations are invalid! Reinitializing population...")
            population.initialize()
            continue
        
        # Evaluate all valid equations
        print("Evaluating equations...")
        eval_results = evaluate_equations(valid_equations_dict, all_benchmarks_data, verbose=True)
        
        if not eval_results:
            print("⚠️  No valid equations in this generation! Reinitializing...")
            population.initialize()
            generations_without_improvement += 1
            continue
        
        # Update fitness scores
        fitness_map = {eq_id: res['avg_correlation'] for eq_id, res in eval_results}
        
        valid_eqs_updated = 0
        for eq in population.equations:
            eq_id = f"ga_gen{eq.generation}_eq{population.equations.index(eq)}"
            if eq_id in fitness_map:
                eq.fitness = fitness_map[eq_id]
                eq.avg_correlation = fitness_map[eq_id]
                # Find r2 score
                for eval_id, eval_res in eval_results:
                    if eval_id == eq_id:
                        eq.avg_r2 = eval_res['avg_r2']
                        break
                valid_eqs_updated += 1
            else:
                # Equation wasn't evaluated (likely invalid), set poor fitness
                eq.fitness = -1.0
                eq.avg_correlation = 0.0
                eq.avg_r2 = -1.0
        
        print(f"  Updated fitness for {valid_eqs_updated}/{len(population.equations)} equations")
        
        # Get statistics
        stats = population.get_statistics()
        
        # Print top 5 equations
        print(f"\nTop 5 equations:")
        for i, (eq_id, res) in enumerate(eval_results[:5], 1):
            # Find the equation object to get simplicity
            eq_obj = next((eq for eq in population.equations if f"ga_gen{eq.generation}_eq{population.equations.index(eq)}" == eq_id), None)
            simplicity = eq_obj.simplicity_score if eq_obj else 0.0
            multi_obj = res['avg_correlation'] - simplicity_weight * simplicity
            
            print(f"  {i}. Corr: {res['avg_correlation']:.6f}, R²: {res['avg_r2']:.6f}, Simplicity: {simplicity:.2f}, Multi-obj: {multi_obj:.6f}")
            print(f"     {res['name'][:90]}")
        
        print(f"\nGeneration statistics:")
        print(f"  Best correlation: {stats['best_correlation']:.6f}")
        print(f"  Avg correlation: {stats['avg_fitness']:.6f}")
        print(f"  Avg complexity: {stats['avg_complexity']:.2f}")
        print(f"  Avg tree depth: {stats['avg_depth']:.1f}")
        print(f"  Diversity: {stats['unique_equations']}/{stats['size']} unique ({stats['diversity']*100:.1f}%)")
        
        # Track best ever
        if stats['best_correlation'] > best_correlation_ever:
            best_correlation_ever = stats['best_correlation']
            generations_without_improvement = 0
            current_mutation_rate = base_mutation_rate  # Reset mutation rate on improvement
            print(f"  🎉 NEW BEST CORRELATION: {best_correlation_ever:.6f}")
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
            population.equations.sort(key=lambda eq: eq.fitness - simplicity_weight * eq.simplicity_score, reverse=True)
            for i in range(n_inject):
                idx = -(i+1)  # Replace from worst
                new_eq = gae.generate_random_equation(4, 5, mandatory_vars)
                new_eq.generation = generation
                population.equations[idx] = new_eq
        
        # Save history
        history.append({
            'generation': generation,
            'best_correlation': stats['best_correlation'],
            'avg_correlation': stats['avg_fitness'],
            'best_r2': stats['best_r2'],
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
        
        print(f"\n{i}. Correlation: {eq.avg_correlation:.6f}, R²: {eq.avg_r2:.6f}, Simplicity: {eq.simplicity_score:.2f}")
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
