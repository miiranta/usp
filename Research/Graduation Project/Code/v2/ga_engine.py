"""
Genetic Algorithm Engine for Equation Evolution

This module implements a genetic algorithm to evolve mathematical equations
that predict benchmark scores based on complexity and parameter count.
"""

import numpy as np
import random
import copy
from typing import List, Dict, Tuple, Callable, Any
from dataclasses import dataclass, field
import ga_building_blocks as gbb

# ============================================================================
# EXPRESSION TREE REPRESENTATION
# ============================================================================

@dataclass
class ExprNode:
    """Node in an expression tree"""
    op: str  # Operation name or 'param' or 'var'
    children: List['ExprNode'] = field(default_factory=list)
    param_idx: int = None  # For parameter nodes
    
    def copy(self):
        """Deep copy of the node"""
        return ExprNode(
            op=self.op,
            children=[child.copy() for child in self.children],
            param_idx=self.param_idx
        )
    
    def get_complexity(self) -> float:
        """Calculate complexity of this subtree"""
        if self.op in ['complexity', 'count']:
            return 0.0
        elif self.op == 'param':
            return 0.0
        else:
            op_obj = gbb.ALL_OPS.get(self.op)
            if op_obj:
                child_complexity = sum(child.get_complexity() for child in self.children)
                return op_obj.complexity_cost + child_complexity
            return 1.0
    
    def get_depth(self) -> int:
        """Get the depth of this subtree"""
        if not self.children:
            return 1
        return 1 + max(child.get_depth() for child in self.children)
    
    def count_nodes(self) -> int:
        """Count total nodes in this subtree"""
        return 1 + sum(child.count_nodes() for child in self.children)
    
    def contains_variable(self, var_name: str) -> bool:
        """Check if this subtree contains a specific variable"""
        if self.op == var_name:
            return True
        return any(child.contains_variable(var_name) for child in self.children)
    
    def get_all_variables(self) -> set:
        """Get all variables used in this subtree"""
        if self.op in ['complexity', 'count']:
            return {self.op}
        variables = set()
        for child in self.children:
            variables.update(child.get_all_variables())
        return variables
    
    def is_degenerate(self) -> bool:
        """Check if this subtree is degenerate (e.g., x - x, x / x, etc.)"""
        # Check for x - x pattern
        if self.op == 'sub' and len(self.children) == 2:
            left_str = self.children[0].to_string()
            right_str = self.children[1].to_string()
            if left_str == right_str:
                return True
        
        # Check for 0 * x pattern
        if self.op == 'mul' and len(self.children) == 2:
            for child in self.children:
                if child.op == 'param':
                    # Can't check param values here, but flag suspicious patterns
                    pass
        
        # Check if all children are degenerate
        if self.children:
            return all(child.is_degenerate() for child in self.children)
        
        return False
    
    def to_string(self, param_values: List[float] = None) -> str:
        """Convert to human-readable string"""
        if self.op in ['complexity', 'count']:
            return self.op
        elif self.op == 'param':
            if param_values and self.param_idx < len(param_values):
                val = param_values[self.param_idx]
                if abs(val) < 0.001 or abs(val) > 1000:
                    return f"{val:.2e}"
                else:
                    return f"{val:.4f}"
            else:
                return f"p{self.param_idx}"
        else:
            op_obj = gbb.ALL_OPS.get(self.op)
            if not op_obj:
                return f"UNKNOWN({self.op})"
            
            symbol = op_obj.symbol
            
            if op_obj.arity == 0:
                return symbol
            elif op_obj.arity == 1:
                child_str = self.children[0].to_string(param_values)
                if self.op in ['sqrt', 'log', 'log2', 'log10', 'sin', 'cos', 'tan', 'abs', 'exp', 'tanh', 'cbrt']:
                    return f"{symbol}({child_str})"
                elif self.op == 'square':
                    return f"({child_str})²"
                elif self.op == 'cube':
                    return f"({child_str})³"
                elif self.op == 'neg':
                    return f"-({child_str})"
                elif self.op == 'reciprocal':
                    return f"1/({child_str})"
                else:
                    return f"{symbol}({child_str})"
            elif op_obj.arity == 2:
                left = self.children[0].to_string(param_values)
                right = self.children[1].to_string(param_values)
                
                if self.op in ['add', 'sub', 'mul', 'div', 'pow']:
                    return f"({left} {symbol} {right})"
                else:
                    return f"{symbol}({left}, {right})"
            elif op_obj.arity == 3:
                args = [child.to_string(param_values) for child in self.children]
                return f"{symbol}({', '.join(args)})"
            
        return "UNKNOWN"
    
    def to_dict(self) -> Dict:
        """Serialize tree to dictionary for JSON/checkpoint storage"""
        result = {'op': self.op}
        if self.op == 'param':
            result['param_idx'] = self.param_idx
        if self.children:
            result['children'] = [child.to_dict() for child in self.children]
        return result
    
    @staticmethod
    def from_dict(data: Dict) -> 'ExprNode':
        """Deserialize tree from dictionary"""
        op = data['op']
        param_idx = data.get('param_idx', 0)
        children_data = data.get('children', [])
        children = [ExprNode.from_dict(child_data) for child_data in children_data]
        return ExprNode(op=op, param_idx=param_idx, children=children)

# ============================================================================
# EQUATION REPRESENTATION
# ============================================================================

@dataclass
class Equation:
    """Represents a complete equation with its tree and metadata"""
    tree: ExprNode
    num_params: int
    param_initial_guess: List[float] = field(default_factory=list)
    fitness: float = -np.inf
    avg_correlation: float = 0.0
    avg_r2: float = 0.0
    simplicity_score: float = 0.0  # Lower is simpler
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    unique_id: int = field(default_factory=lambda: id(object()))  # Unique identifier
    
    def __post_init__(self):
        """Initialize parameter initial guesses if not provided"""
        if not self.param_initial_guess and self.num_params > 0:
            self.param_initial_guess = [gbb.generate_random_constant() for _ in range(self.num_params)]
        # Calculate simplicity score
        self.simplicity_score = self.get_complexity() + 0.1 * self.get_depth() + 0.05 * self.get_size()
    
    def get_id(self) -> str:
        """Get a unique ID for this equation"""
        return f"gen{self.generation}_{self.to_string()[:50]}"
    
    def to_string(self, param_values: List[float] = None) -> str:
        """Convert to human-readable string"""
        return self.tree.to_string(param_values)
    
    def to_lambda(self) -> Callable:
        """Convert tree to executable lambda function"""
        def evaluate(node: ExprNode, complexity, count, params):
            if node.op == 'complexity':
                return complexity
            elif node.op == 'count':
                return count
            elif node.op == 'param':
                # Bounds check to prevent IndexError
                if node.param_idx >= len(params):
                    return 1.0  # Return default value if param index out of bounds
                return params[node.param_idx]
            else:
                op_obj = gbb.ALL_OPS.get(node.op)
                if not op_obj:
                    raise ValueError(f"Unknown operation: {node.op}")
                
                child_values = [evaluate(child, complexity, count, params) for child in node.children]
                return op_obj.func(*child_values)
        
        return lambda complexity, count, *params: evaluate(self.tree, complexity, count, params)
    
    def get_complexity(self) -> float:
        """Get the complexity score of this equation"""
        return self.tree.get_complexity()
    
    def get_depth(self) -> int:
        """Get the depth of the expression tree"""
        return self.tree.get_depth()
    
    def get_size(self) -> int:
        """Get the total number of nodes"""
        return self.tree.count_nodes()
    
    def contains_variable(self, var_name: str) -> bool:
        """Check if equation uses a specific variable"""
        return self.tree.contains_variable(var_name)
    
    def get_all_variables(self) -> set:
        """Get all variables used in this equation"""
        return self.tree.get_all_variables()
    
    def is_degenerate(self) -> bool:
        """Check if equation is degenerate (e.g., always zero, always constant)"""
        return self.tree.is_degenerate()
    
    def is_valid(self) -> bool:
        """Check if equation is valid (not degenerate, uses required variables)"""
        # Must not be degenerate
        if self.is_degenerate():
            return False
        
        # Must use at least one variable
        vars_used = self.get_all_variables()
        if len(vars_used) == 0:
            return False
        
        # Check string representation for obvious problems
        eq_str = self.to_string()
        if eq_str == '0' or eq_str == '0.0':
            return False
        
        return True
    
    def get_canonical_form(self) -> str:
        """Get a canonical string representation for duplicate detection"""
        return self.to_string()
    
    def copy(self):
        """Create a deep copy of this equation"""
        return Equation(
            tree=self.tree.copy(),
            num_params=self.num_params,
            param_initial_guess=self.param_initial_guess.copy(),
            fitness=self.fitness,
            avg_correlation=self.avg_correlation,
            avg_r2=self.avg_r2,
            simplicity_score=self.simplicity_score,
            generation=self.generation,
            parent_ids=self.parent_ids.copy()
        )

# ============================================================================
# EQUATION GENERATION
# ============================================================================

def generate_random_tree(max_depth: int, current_depth: int, num_params: int, 
                        mandatory_vars: List[str] = None, current_vars: set = None) -> ExprNode:
    """Generate a random expression tree, ensuring mandatory variables are included"""
    
    if current_vars is None:
        current_vars = set()
    
    # Terminal probability increases with depth
    terminal_prob = current_depth / max_depth
    
    # Check if we need to force a mandatory variable at this level
    if mandatory_vars and current_depth < max_depth - 1:
        missing_vars = set(mandatory_vars) - current_vars
        if missing_vars and random.random() < 0.3:  # 30% chance to add missing var
            var = random.choice(list(missing_vars))
            current_vars.add(var)
            return ExprNode(op=var)
    
    if current_depth >= max_depth or (current_depth > 1 and random.random() < terminal_prob):
        # Generate terminal
        choice = random.random()
        if choice < 0.4:
            current_vars.add('complexity')
            return ExprNode(op='complexity')
        elif choice < 0.8:
            current_vars.add('count')
            return ExprNode(op='count')
        else:
            return ExprNode(op='param', param_idx=random.randint(0, num_params - 1))
    
    # Generate non-terminal
    # Favor binary operations for more interesting equations
    arity_choice = random.random()
    if arity_choice < 0.1:
        arity = 1
    elif arity_choice < 0.85:
        arity = 2
    else:
        arity = 3
    
    ops = gbb.get_operations_by_arity(arity)
    if not ops:
        # Fallback to terminal
        current_vars.add('complexity')
        return ExprNode(op='complexity')
    
    op_name = random.choice(list(ops.keys()))
    children = [generate_random_tree(max_depth, current_depth + 1, num_params, mandatory_vars, current_vars) 
                for _ in range(arity)]
    
    return ExprNode(op=op_name, children=children)

def generate_simple_equation(num_params: int = 3) -> Equation:
    """Generate a simple starting equation"""
    # Common patterns:
    # a * complexity^b * count^c
    # a * complexity + b * count + c
    # a * sqrt(complexity) * count^b + c
    
    pattern = random.choice(['power', 'linear', 'mixed'])
    
    if pattern == 'power':
        # a * complexity^b * count^c
        tree = ExprNode(op='mul', children=[
            ExprNode(op='param', param_idx=0),
            ExprNode(op='mul', children=[
                ExprNode(op='pow', children=[
                    ExprNode(op='complexity'),
                    ExprNode(op='param', param_idx=1)
                ]),
                ExprNode(op='pow', children=[
                    ExprNode(op='count'),
                    ExprNode(op='param', param_idx=2)
                ])
            ])
        ])
        return Equation(tree=tree, num_params=3, param_initial_guess=[1.0, 0.3, 1.4])
    
    elif pattern == 'linear':
        # a * complexity + b * count + c
        tree = ExprNode(op='add', children=[
            ExprNode(op='add', children=[
                ExprNode(op='mul', children=[
                    ExprNode(op='param', param_idx=0),
                    ExprNode(op='complexity')
                ]),
                ExprNode(op='mul', children=[
                    ExprNode(op='param', param_idx=1),
                    ExprNode(op='count')
                ])
            ]),
            ExprNode(op='param', param_idx=2)
        ])
        return Equation(tree=tree, num_params=3, param_initial_guess=[1.0, 1.0, 0.0])
    
    else:  # mixed
        # a * sqrt(complexity) * count^b + c
        tree = ExprNode(op='add', children=[
            ExprNode(op='mul', children=[
                ExprNode(op='param', param_idx=0),
                ExprNode(op='mul', children=[
                    ExprNode(op='sqrt', children=[ExprNode(op='complexity')]),
                    ExprNode(op='pow', children=[
                        ExprNode(op='count'),
                        ExprNode(op='param', param_idx=1)
                    ])
                ])
            ]),
            ExprNode(op='param', param_idx=2)
        ])
        return Equation(tree=tree, num_params=3, param_initial_guess=[1.0, 1.4, 0.0])

def generate_random_equation(max_depth: int = 4, num_params: int = 4, 
                           mandatory_vars: List[str] = None) -> Equation:
    """Generate a completely random equation, ensuring mandatory variables are included"""
    
    # Generate tree with mandatory variables
    current_vars = set()
    tree = generate_random_tree(max_depth, 0, num_params, mandatory_vars, current_vars)
    
    # Verify mandatory variables are present
    if mandatory_vars:
        eq = Equation(tree=tree, num_params=num_params)
        for var in mandatory_vars:
            if not eq.contains_variable(var):
                # Force add the variable by creating a multiplication with it
                tree = ExprNode(op='mul', children=[
                    tree,
                    ExprNode(op=var)
                ])
                break
    
    return Equation(tree=tree, num_params=num_params)

# ============================================================================
# GENETIC OPERATORS
# ============================================================================

def mutate_subtree(node: ExprNode, num_params: int, mutation_rate: float = 0.3) -> ExprNode:
    """Mutate a subtree with given probability"""
    
    if random.random() < mutation_rate:
        # Replace this subtree with a new random one
        max_depth = random.randint(1, 3)
        return generate_random_tree(max_depth, 0, num_params)
    
    # Recursively mutate children
    new_children = [mutate_subtree(child, num_params, mutation_rate * 0.7) for child in node.children]
    new_node = node.copy()
    new_node.children = new_children
    
    return new_node

def crossover(parent1: Equation, parent2: Equation) -> Tuple[Equation, Equation]:
    """Perform crossover between two equations"""
    
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    # Select random subtrees from each parent
    def get_random_node(node: ExprNode, depth: int = 0) -> Tuple[ExprNode, int]:
        """Get a random node from the tree, returning (node, depth)"""
        all_nodes = []
        
        def collect_nodes(n: ExprNode, d: int):
            all_nodes.append((n, d))
            for child in n.children:
                collect_nodes(child, d + 1)
        
        collect_nodes(node, depth)
        return random.choice(all_nodes) if all_nodes else (node, depth)
    
    # Get random nodes from both trees
    node1, depth1 = get_random_node(child1.tree)
    node2, depth2 = get_random_node(child2.tree)
    
    # Swap the subtrees (simplified - just copy attributes)
    # In a full implementation, we'd need to track parent pointers
    if node1.children and node2.children and len(node1.children) == len(node2.children):
        node1.children, node2.children = node2.children, node1.children
    
    child1.parent_ids = [parent1.get_id(), parent2.get_id()]
    child2.parent_ids = [parent1.get_id(), parent2.get_id()]
    
    return child1, child2

def mutate_equation(equation: Equation, mutation_rate: float = 0.2) -> Equation:
    """Mutate an equation"""
    
    max_attempts = 10
    for attempt in range(max_attempts):
        mutated = equation.copy()
        
        # Mutate the tree structure
        mutated.tree = mutate_subtree(mutated.tree, mutated.num_params, mutation_rate)
        
        # Mutate parameter initial guesses
        for i in range(len(mutated.param_initial_guess)):
            if random.random() < mutation_rate:
                mutated.param_initial_guess[i] += random.gauss(0, 0.5)
        
        # Add a new parameter sometimes
        if random.random() < mutation_rate * 0.5 and mutated.num_params < 8:
            mutated.num_params += 1
            mutated.param_initial_guess.append(gbb.generate_random_constant())
        
        mutated.parent_ids = [equation.get_id()]
        
        # Recalculate simplicity
        mutated.simplicity_score = mutated.get_complexity() + 0.1 * mutated.get_depth() + 0.05 * mutated.get_size()
        
        # Validate before returning
        if mutated.is_valid():
            return mutated
    
    # If all attempts failed, return a random equation
    return generate_random_equation(max_depth=4, num_params=equation.num_params)

def simplify_equation(equation: Equation) -> Equation:
    """Attempt to simplify an equation"""
    
    def simplify_node(node: ExprNode) -> ExprNode:
        """Simplify a node recursively"""
        
        # First simplify children
        new_children = [simplify_node(child) for child in node.children]
        new_node = node.copy()
        new_node.children = new_children
        
        # Apply simplification rules
        if node.op == 'mul':
            # If multiplying by a parameter that's close to 1, might remove it
            pass
        elif node.op == 'add':
            # If adding a parameter that's close to 0, might remove it
            pass
        
        return new_node
    
    simplified = equation.copy()
    simplified.tree = simplify_node(simplified.tree)
    simplified.parent_ids = [equation.get_id()]
    
    return simplified

# ============================================================================
# POPULATION MANAGEMENT
# ============================================================================

class Population:
    """Manages a population of equations"""
    
    def __init__(self, size: int, max_depth: int = 4, num_params: int = 4, 
                 mandatory_vars: List[str] = None, simplicity_weight: float = 0.1):
        self.size = size
        self.max_depth = max_depth
        self.num_params = num_params
        self.mandatory_vars = mandatory_vars or []
        self.simplicity_weight = simplicity_weight  # Weight for multi-objective optimization
        self.equations: List[Equation] = []
        self.generation = 0
        self.best_ever: Equation = None
    
    def initialize(self, initial_equations: List[Equation] = None):
        """Initialize the population with random equations or from checkpoint"""
        self.equations = []
        
        if initial_equations:
            # Resume from checkpoint
            print(f"  Loading {len(initial_equations)} equations from checkpoint...")
            self.equations = [eq.copy() for eq in initial_equations[:self.size]]
            # Update generation number
            if self.equations:
                self.generation = max(eq.generation for eq in self.equations)
                print(f"  Resuming from generation {self.generation}")
        
        # Fill remaining slots with new equations
        remaining = self.size - len(self.equations)
        if remaining > 0:
            print(f"  Generating {remaining} new equations...")
            
            # Start with some simple equations
            for _ in range(min(remaining // 3, remaining)):
                eq = generate_simple_equation(self.num_params)
                # Verify mandatory variables
                if self.mandatory_vars:
                    for var in self.mandatory_vars:
                        if not eq.contains_variable(var):
                            # Add the variable
                            eq.tree = ExprNode(op='mul', children=[eq.tree, ExprNode(op=var)])
                            eq.simplicity_score = eq.get_complexity() + 0.1 * eq.get_depth() + 0.05 * eq.get_size()
                eq.generation = self.generation
                self.equations.append(eq)
            
            # Add random equations for the rest
            for _ in range(self.size - len(self.equations)):
                eq = generate_random_equation(self.max_depth, self.num_params, self.mandatory_vars)
                eq.generation = self.generation
                self.equations.append(eq)
    
    def select_parents(self, tournament_size: int = 3) -> Tuple[Equation, Equation]:
        """Select two parents using tournament selection with multi-objective fitness"""
        
        def tournament():
            candidates = random.sample(self.equations, min(tournament_size, len(self.equations)))
            # Multi-objective: maximize correlation, minimize simplicity
            return max(candidates, key=lambda eq: eq.fitness - self.simplicity_weight * eq.simplicity_score)
        
        return tournament(), tournament()
    
    def evolve(self, elite_size: int = 5, crossover_rate: float = 0.7, mutation_rate: float = 0.3, 
               stagnation_counter: int = 0):
        """Evolve the population to the next generation with adaptive parameters"""
        
        # Sort by multi-objective fitness (correlation - simplicity_penalty)
        self.equations.sort(key=lambda eq: eq.fitness - self.simplicity_weight * eq.simplicity_score, reverse=True)
        
        # Update best ever
        if not self.best_ever or self.equations[0].fitness > self.best_ever.fitness:
            self.best_ever = self.equations[0].copy()
        
        # Adaptive parameters based on stagnation
        if stagnation_counter > 3:
            # Increase mutation and reduce crossover to escape local optimum
            mutation_rate = min(0.6, mutation_rate * 1.5)
            crossover_rate = max(0.4, crossover_rate * 0.8)
            elite_size = max(2, elite_size - 1)  # Reduce elite to allow more diversity
            print(f"    Adapting: mutation_rate={mutation_rate:.2f}, crossover_rate={crossover_rate:.2f}, elite_size={elite_size}")
        
        # Keep elite (but validate them)
        new_population = []
        for eq in self.equations[:elite_size]:
            if eq.is_valid():
                new_population.append(eq.copy())
        
        # Add some completely random equations if stagnating badly
        if stagnation_counter > 5:
            num_random = min(self.size // 5, 10)
            print(f"    Injecting {num_random} random equations to break stagnation")
            for _ in range(num_random):
                random_eq = generate_random_equation(max_depth=4, num_params=5, mandatory_vars=self.mandatory_vars)
                random_eq.generation = self.generation + 1
                new_population.append(random_eq)
        
        # Generate offspring with validation
        attempts = 0
        max_attempts = self.size * 10  # Prevent infinite loops
        seen_forms = {eq.get_canonical_form() for eq in new_population}  # Track duplicates
        
        while len(new_population) < self.size and attempts < max_attempts:
            attempts += 1
            
            parent1, parent2 = self.select_parents()
            
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            if random.random() < mutation_rate:
                child1 = mutate_equation(child1, mutation_rate)
            
            if random.random() < mutation_rate:
                child2 = mutate_equation(child2, mutation_rate)
            
            child1.generation = self.generation + 1
            child2.generation = self.generation + 1
            
            # Only add valid and non-duplicate children
            child1_form = child1.get_canonical_form()
            if child1.is_valid() and child1_form not in seen_forms:
                new_population.append(child1)
                seen_forms.add(child1_form)
            
            child2_form = child2.get_canonical_form()
            if len(new_population) < self.size and child2.is_valid() and child2_form not in seen_forms:
                new_population.append(child2)
                seen_forms.add(child2_form)
        
        # If we still don't have enough, fill with random equations
        random_attempts = 0
        while len(new_population) < self.size and random_attempts < max_attempts:
            random_attempts += 1
            random_eq = generate_random_equation(max_depth=4, num_params=5, mandatory_vars=self.mandatory_vars)
            random_eq.generation = self.generation + 1
            random_form = random_eq.get_canonical_form()
            if random_eq.is_valid() and random_form not in seen_forms:
                new_population.append(random_eq)
                seen_forms.add(random_form)
        
        # Last resort: if still not enough, allow duplicates but force mutation
        while len(new_population) < self.size:
            random_eq = generate_random_equation(max_depth=4, num_params=5, mandatory_vars=self.mandatory_vars)
            random_eq.generation = self.generation + 1
            if random_eq.is_valid():
                new_population.append(random_eq)
        
        self.equations = new_population[:self.size]
        self.generation += 1
        
        # Check diversity
        unique_forms = len({eq.get_canonical_form() for eq in self.equations})
        if unique_forms < self.size * 0.5:
            print(f"    WARNING: Low diversity! Only {unique_forms}/{self.size} unique equations")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get population statistics"""
        fitnesses = [eq.fitness for eq in self.equations if eq.fitness > -np.inf]
        complexities = [eq.get_complexity() for eq in self.equations]
        depths = [eq.get_depth() for eq in self.equations]
        sizes = [eq.get_size() for eq in self.equations]
        
        # Calculate diversity
        unique_forms = len({eq.get_canonical_form() for eq in self.equations})
        diversity = unique_forms / len(self.equations) if self.equations else 0.0
        
        return {
            'generation': self.generation,
            'size': len(self.equations),
            'unique_equations': unique_forms,
            'diversity': diversity,
            'best_fitness': max(fitnesses) if fitnesses else -np.inf,
            'avg_fitness': np.mean(fitnesses) if fitnesses else -np.inf,
            'worst_fitness': min(fitnesses) if fitnesses else -np.inf,
            'avg_complexity': np.mean(complexities),
            'avg_depth': np.mean(depths),
            'avg_size': np.mean(sizes),
            'best_correlation': self.equations[0].avg_correlation if self.equations else 0.0,
        }

# ============================================================================
# MAIN INTERFACE
# ============================================================================

if __name__ == '__main__':
    # Test equation generation
    print("Testing equation generation...")
    
    for i in range(5):
        eq = generate_random_equation(max_depth=3, num_params=4)
        print(f"\nEquation {i+1}:")
        print(f"  Expression: {eq.to_string()}")
        print(f"  Complexity: {eq.get_complexity():.2f}")
        print(f"  Depth: {eq.get_depth()}")
        print(f"  Size: {eq.get_size()}")
