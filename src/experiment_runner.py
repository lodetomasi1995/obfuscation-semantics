"""
Experiment Runner for LLM-Driven Code Obfuscation
=================================================
This module provides a framework for running experiments to evaluate
different LLMs on code obfuscation tasks with the Semantic Elasticity metric.

Author: Lorenzo De Tomasi
Institution: Università degli Studi dell'Aquila
"""

import os
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Any, Tuple, Optional
import importlib.util
import tempfile
import sys
import traceback
from scipy import stats

# Import local modules
from code_obfuscator import CodeObfuscator, SAMPLE_FUNCTIONS
from semantic_elasticity import SemanticElasticityCalculator

# Define function categories for organized evaluation
FUNCTION_CATEGORIES = {
    "Mathematical": [
        "factorial", "fibonacci", "is_prime", "gcd", 
        "lcm", "power", "sqrt_newton"
    ],
    "Sorting & Searching": [
        "bubble_sort", "binary_search", "merge_sort", "quick_sort", 
        "insertion_sort", "linear_search"
    ],
    "String Manipulation": [
        "str_reverse", "is_palindrome", "word_count", "longest_common_substring",
        "levenshtein_distance", "count_vowels"
    ],
    "Data Structure": [
        "flatten_list", "list_permutations", "dict_merge", 
        "remove_duplicates", "rotate_array"
    ],
    "Recursive": [
        "tower_of_hanoi", "binary_tree_depth", "flood_fill", 
        "knapsack", "edit_distance", "coin_change"
    ]
}

def generate_test_cases(function_name):
    """Generate comprehensive test suites with edge cases for robust function validation"""
    test_cases = {
        # Mathematical functions with edge cases and corner cases
        "factorial": [
            0,                  # Edge case: factorial of 0
            1,                  # Edge case: factorial of 1
            5,                  # Typical case
            10,                 # Medium value
            12,                 # Large value for reasonable computation
            # Edge cases to test robustness (functions should handle these appropriately)
            -1,                 # Invalid input: negative
            4.5,                # Invalid input: non-integer
            "5"                 # Invalid input: string representation of number
        ],
        
        "fibonacci": [
            0,                  # Edge case: fibonacci(0) = 0
            1,                  # Edge case: fibonacci(1) = 1
            2,                  # Edge case: fibonacci(2) = 1
            5,                  # Typical small case
            10,                 # Medium case
            15,                 # Larger value
            20,                 # Performance test
            -1,                 # Invalid input: negative
            1.5                 # Invalid input: non-integer
        ],
        
        "is_prime": [
            0,                  # Edge case: not prime
            1,                  # Edge case: not prime by definition
            2,                  # Edge case: smallest prime
            3,                  # Small prime
            4,                  # Composite
            7,                  # Typical prime
            9,                  # Perfect square composite
            11,                 # Prime
            13,                 # Prime
            15,                 # Multiple of small primes
            17,                 # Prime
            20,                 # Even composite
            23,                 # Prime
            25,                 # Perfect square
            121,                # Large perfect square
            997,                # Large prime
            1000,               # Large composite
            -7,                 # Invalid input: negative
            2.5                 # Invalid input: non-integer
        ],
        
        "gcd": [
            (10, 5),            # Basic case, one divides the other
            (17, 23),           # Coprime numbers
            (0, 5),             # Edge case: one is zero
            (5, 0),             # Edge case: other is zero
            (0, 0),             # Edge case: both zero
            (48, 18),           # Common factors
            (7, 13),            # Coprime
            (100, 10),          # Power of 10
            (-48, 18),          # Negative value
            (48, -18),          # Negative value
            (-48, -18),         # Both negative
            (2**31-1, 2**30-1)  # Large values
        ],
        
        # Sorting and searching algorithms with diverse data patterns
        "bubble_sort": [
            [1, 2, 3, 4, 5],                # Already sorted
            [5, 4, 3, 2, 1],                # Reverse sorted
            [3, 1, 4, 1, 5, 9, 2, 6],       # Random order with duplicates
            [9, 8, 7, 6, 5, 4, 3, 2, 1, 0], # Perfectly reversed
            [],                             # Empty array
            [1],                            # Single element
            [1, 1, 1, 1, 1],                # All identical
            ["a", "c", "b"],                # Strings
            [0.1, 0.5, 0.2, 0.9, 0.3]       # Floating point
        ],
        
        "binary_search": [
            ([1, 2, 3, 4, 5], 1),           # Find first element
            ([1, 2, 3, 4, 5], 3),           # Find middle element
            ([1, 2, 3, 4, 5], 5),           # Find last element
            ([1, 2, 3, 4, 5], 0),           # Element smaller than all
            ([1, 2, 3, 4, 5], 6),           # Element larger than all
            ([1, 2, 3, 4, 5], 2.5),         # Element between values
            ([1, 3, 5, 7, 9], 5),           # In odd-length array
            ([2, 4, 6, 8], 6),              # In even-length array
            ([1, 1, 1, 1, 1], 1),           # All identical, target exists
            ([1, 1, 1, 1, 1], 2),           # All identical, target doesn't exist
            ([], 5),                        # Empty array
            ([5], 5),                       # Single element, found
            ([5], 7),                       # Single element, not found
            ([1, 3, 5, 7, 9, 11, 13, 15, 17, 19], 15), # Larger array
            (list(range(0, 10000, 10)), 5000)  # Very large array
        ],
        
        # String manipulation with complex test cases
        "str_reverse": [
            "",                             # Empty string
            "a",                            # Single character
            "hello",                        # Basic word
            "racecar",                      # Palindrome
            "Python",                       # Mixed case
            "A man, a plan, a canal: Panama", # With punctuation
            "The quick brown fox jumps over the lazy dog", # Long sentence
            "   spaced   ",                 # With whitespace
            "123!@#",                       # Alphanumeric with special chars
            "😊🙂🥳",                          # Unicode emojis
            "Löwe 老虎 Léopard",             # International characters
            "\t\n\r"                        # Escape sequences
        ],
        
        "is_palindrome": [
            "",                             # Empty string (trivial palindrome)
            "a",                            # Single character (trivial palindrome)
            "hello",                        # Not a palindrome
            "racecar",                      # Perfect palindrome
            "A man, a plan, a canal: Panama", # Palindrome with spaces and punctuation
            "Not a palindrome",             # Clearly not a palindrome
            "No lemon, no melon",           # Palindrome with spaces and commas
            "Mr. Owl ate my metal worm",    # Palindrome ignoring spaces and punctuation
            "Was it a car or a cat I saw?", # Palindrome with question mark
            "Doc, note: I dissent. A fast never prevents a fatness. I diet on cod.", # Complex palindrome
            "Madam, I'm Adam",              # Palindrome with apostrophe
            "Step on no pets",              # Word-by-word palindrome
            "12321",                        # Numeric palindrome
            "123 321",                      # Numeric palindrome with space
            "Not\ta\tpalindrome"            # With tab characters
        ]
    }
    
    # Add more test cases as needed
    
    return test_cases.get(function_name, [])

def test_function(orig_code: str, obfs_code: str, test_cases: List[Any]) -> Dict:
    """Test functional equivalence and collect metrics for evaluation"""
    try:
        # Skip testing if obfuscated code is None (failed to generate)
        if obfs_code is None:
            return {
                'metrics': {
                    'pass_rate': 0.0,
                    'code_expansion': 0.0,
                    'cyclomatic_complexity': {'original': 0, 'obfuscated': 0},
                    'identifier_entropy': {'original': 0, 'obfuscated': 0},
                    'avg_timing_diff_ms': 0.0,
                    'semantic_elasticity': 0.0
                }
            }

        # Initialize metrics calculators
        obfuscator = CodeObfuscator()
        se_calculator = SemanticElasticityCalculator()

        # Calculate basic metrics
        cc_original = obfuscator.compute_cyclomatic_complexity(orig_code)
        cc_obfuscated = obfuscator.compute_cyclomatic_complexity(obfs_code)
        
        ie_original = obfuscator.compute_identifier_entropy(orig_code)
        ie_obfuscated = obfuscator.compute_identifier_entropy(obfs_code)
        
        code_expansion = len(obfs_code.split('\n')) / len(orig_code.split('\n'))
        
        metrics = {
            'cyclomatic_complexity': {
                'original': cc_original,
                'obfuscated': cc_obfuscated
            },
            'identifier_entropy': {
                'original': ie_original,
                'obfuscated': ie_obfuscated
            },
            'code_expansion': code_expansion
        }

        # If no test cases, we can't test functionality
        if not test_cases:
            metrics['pass_rate'] = 0.0
            metrics['avg_timing_diff_ms'] = 0.0
            metrics['semantic_elasticity'] = 0.0
            return {'metrics': metrics}

        # Load modules for testing
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as f:
            f.write(orig_code)
            orig_path = f.name
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as f:
            f.write(obfs_code)
            obfs_path = f.name

        orig_spec = importlib.util.spec_from_file_location("original", orig_path)
        obfs_spec = importlib.util.spec_from_file_location("obfuscated", obfs_path)

        orig_module = importlib.util.module_from_spec(orig_spec)
        obfs_module = importlib.util.module_from_spec(obfs_spec)

        sys.modules["original"] = orig_module
        sys.modules["obfuscated"] = obfs_module

        orig_spec.loader.exec_module(orig_module)
        obfs_spec.loader.exec_module(obfs_module)

        # Find function names
        orig_func_name = None
        obfs_func_name = None
        
        for node in ast.walk(ast.parse(orig_code)):
            if isinstance(node, ast.FunctionDef):
                orig_func_name = node.name
                break

        for node in ast.walk(ast.parse(obfs_code)):
            if isinstance(node, ast.FunctionDef):
                obfs_func_name = node.name
                break
                
        if not orig_func_name or not obfs_func_name:
            # Clean up temporary files
            try:
                os.unlink(orig_path)
                os.unlink(obfs_path)
            except:
                pass
            return {'error': 'Function definition not found', 'metrics': {'pass_rate': 0.0}}

        # Get functions
        orig_func = getattr(orig_module, orig_func_name)
        obfs_func = getattr(obfs_module, obfs_func_name)

        # Run tests
        results = []
        passed = 0
        timing_diffs = []

        for test_input in test_cases:
            try:
                # Measure timing for original
                start = time.perf_counter()
                if isinstance(test_input, tuple):
                    orig_result = orig_func(*test_input)
                else:
                    orig_result = orig_func(test_input)
                orig_time = time.perf_counter() - start

                # Measure timing for obfuscated
                start = time.perf_counter()
                if isinstance(test_input, tuple):
                    obfs_result = obfs_func(*test_input)
                else:
                    obfs_result = obfs_func(test_input)
                obfs_time = time.perf_counter() - start

                # Check if results are equal
                equal = orig_result == obfs_result
                if equal:
                    passed += 1

                timing_diffs.append(obfs_time - orig_time)

                results.append({
                    'input': test_input,
                    'original_output': orig_result,
                    'obfuscated_output': obfs_result,
                    'equal': equal,
                    'timing_diff_ms': (obfs_time - orig_time) * 1000
                })
            except Exception as e:
                results.append({
                    'input': test_input,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })

        # Clean up temporary files
        try:
            os.unlink(orig_path)
            os.unlink(obfs_path)
        except:
            pass

        # Calculate pass rate and average timing difference
        pass_rate = passed / len(test_cases) if test_cases else 0
        avg_timing_diff_ms = (sum(timing_diffs) / len(timing_diffs) * 1000) if timing_diffs else 0

        # Update metrics with test results
        metrics.update({
            'pass_rate': pass_rate,
            'avg_timing_diff_ms': avg_timing_diff_ms
        })
        
        # Calculate Semantic Elasticity
        se_score = se_calculator.calculate(orig_code, obfs_code, pass_rate)
        metrics['semantic_elasticity'] = se_score

        return {
            'results': results,
            'metrics': metrics,
            'original_code': orig_code,
            'obfuscated_code': obfs_code
        }

    except Exception as e:
        print(f"Error in testing: {e}")
        print(traceback.format_exc())
        return {'error': str(e), 'metrics': {'pass_rate': 0.0}}


class ExperimentRunner:
    """Framework for running code obfuscation experiments with different LLMs"""
    
    def __init__(self, config_file=None):
        """
        Initialize the experiment runner
        
        Args:
            config_file: Path to JSON configuration file (optional)
        """
        self.config = {
            "models": ["claude", "gemini", "gpt-4"],
            "approaches": ["standard", "few-shot"],
            "trials": 3,
            "functions": ["factorial", "fibonacci", "binary_search", "str_reverse"],
            "output_dir": "results",
            "anthropic_api_key": None,
            "gemini_api_key": None,
            "openai_api_key": None
        }
        
        # Load configuration from file if provided
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                    self.config.update(loaded_config)
            except Exception as e:
                print(f"Error loading configuration file: {e}")
                
        # Create output directory if it doesn't exist
        os.makedirs(self.config["output_dir"], exist_ok=True)
        os.makedirs(os.path.join(self.config["output_dir"], "data"), exist_ok=True)
        os.makedirs(os.path.join(self.config["output_dir"], "figures"), exist_ok=True)
        
        # Initialize the obfuscator
        self.obfuscator = CodeObfuscator(
            anthropic_api_key=self.config.get("anthropic_api_key"),
            gemini_api_key=self.config.get("gemini_api_key"),
            openai_api_key=self.config.get("openai_api_key")
        )
        
        # Initialize semantic elasticity calculator
        self.se_calculator = SemanticElasticityCalculator()
        
    def run(self):
        """Run the configured experiments"""
        print("Starting Code Obfuscation Experiments")
        print("=" * 50)
        
        # Get configuration
        models = self.config["models"]
        approaches = self.config["approaches"]
        trials = self.config["trials"]
        functions = self.config["functions"]
        
        # Summary
        print(f"Models: {', '.join(models)}")
        print(f"Approaches: {', '.join(approaches)}")
        print(f"Functions: {len(functions)}")
        print(f"Trials per combination: {trials}")
        total_experiments = len(models) * len(approaches) * len(functions) * trials
        print(f"Total experiments: {total_experiments}")
        print("=" * 50)
        
        # Results containers
        all_results = []
        
        # Progress tracking
        progress_bar = tqdm(total=total_experiments, desc="Running experiments")
        
        # Run experiments for each combination
        for function_name in functions:
            print(f"\nProcessing function: {function_name}")
            
            # Get original code
            orig_code = SAMPLE_FUNCTIONS.get(function_name, "")
            if not orig_code:
                print(f"Warning: No code found for {function_name}")
                progress_bar.update(len(models) * len(approaches) * trials)
                continue
            
            # Get test cases
            test_cases = generate_test_cases(function_name)
            
            for model in models:
                for approach in approaches:
                    for trial in range(trials):
                        try:
                            # Time the obfuscation process
                            start_time = time.time()
                            
                            # Apply obfuscation based on approach
                            if approach == "standard":
                                obfuscated_code = self.obfuscator.obfuscate_code(orig_code, model_name=model)
                            else:  # few-shot
                                obfuscated_code = self.obfuscator.obfuscate_code_with_examples(orig_code, model_name=model)
                                
                            obfuscation_time = time.time() - start_time
                            
                            # Test functionality
                            test_result = test_function(orig_code, obfuscated_code, test_cases)
                            
                            if 'error' in test_result:
                                print(f"  Error in {model}/{approach}/{function_name} trial {trial+1}: {test_result['error']}")
                                metrics = {'pass_rate': 0.0}
                            else:
                                metrics = test_result['metrics']
                            
                            # Add extra information
                            metrics['obfuscation_time'] = obfuscation_time
                            metrics['code_lines'] = len(obfuscated_code.split('\n')) if obfuscated_code else 0
                            metrics['code_chars'] = len(obfuscated_code) if obfuscated_code else 0
                            metrics['trial'] = trial + 1
                            metrics['function'] = function_name
                            metrics['model'] = model
                            metrics['approach'] = approach
                            
                            # Calculate differences for easier analysis
                            if 'cyclomatic_complexity' in metrics:
                                metrics['cc_diff'] = metrics['cyclomatic_complexity']['obfuscated'] - metrics['cyclomatic_complexity']['original']
                            else:
                                metrics['cc_diff'] = 0
                                
                            if 'identifier_entropy' in metrics:
                                metrics['ie_diff'] = metrics['identifier_entropy']['obfuscated'] - metrics['identifier_entropy']['original']
                            else:
                                metrics['ie_diff'] = 0
                                
                            # Add to results
                            all_results.append(metrics)
                            
                            # Save individual result
                            result_filename = f"{function_name}_{model}_{approach}_trial{trial+1}.json"
                            with open(os.path.join(self.config["output_dir"], "data", result_filename), 'w') as f:
                                json.dump({
                                    'metrics': metrics,
                                    'original_code': orig_code,
                                    'obfuscated_code': obfuscated_code
                                }, f, indent=2)
                            
                            # Print brief summary
                            print(f"  {model}/{approach} Trial {trial+1}: " + 
                                 f"Pass rate: {metrics.get('pass_rate', 0)*100:.1f}%, " +
                                 f"SE: {metrics.get('semantic_elasticity', 0):.2f}")
                            
                        except Exception as e:
                            print(f"  Error in {model}/{approach}/{function_name} trial {trial+1}: {e}")
                            print(f"  {traceback.format_exc()}")
                            
                        progress_bar.update(1)
        
        progress_bar.close()
        
        # Convert to DataFrame
        df_results = pd.DataFrame(all_results)
        
        # Save combined results
        df_results.to_csv(os.path.join(self.config["output_dir"], "obfuscation_results.csv"), index=False)
        
        print(f"\nExperiments completed. Results saved to {self.config['output_dir']}")
        
        # Analyze results
        self.analyze_results(df_results)
        
        return df_results
        
    def analyze_results(self, df):
        """Analyze experimental results and generate visualizations"""
        print("\nAnalyzing results...")
        
        # Add function category information
        def get_function_category(func_name):
            for category, functions in FUNCTION_CATEGORIES.items():
                if func_name in functions:
                    return category
            return "Other"
        
        df['category'] = df['function'].apply(get_function_category)
        
        # Save the augmented DataFrame
        df.to_csv(os.path.join(self.config["output_dir"], "analyzed_results.csv"), index=False)
        
        # Overall metrics by model and approach
        model_approach_metrics = df.groupby(['model', 'approach']).agg({
            'pass_rate': ['mean', 'std'],
            'code_expansion': ['mean', 'std'],
            'cc_diff': ['mean', 'std'],
            'ie_diff': ['mean', 'std'],
            'semantic_elasticity': ['mean', 'max'],
            'obfuscation_time': ['mean', 'std']
        }).reset_index()
        
        # Save summary metrics
        model_approach_metrics.to_csv(os.path.join(self.config["output_dir"], "model_approach_summary.csv"))
        
        print("\nModel performance by approach:")
        print(model_approach_metrics)
        
        # Generate visualizations
        self._create_basic_visualizations(df)
        self._create_advanced_visualizations(df)
        self._create_semantic_elasticity_visualizations(df)
        
        print(f"Analysis complete. Visualizations saved to {os.path.join(self.config['output_dir'], 'figures')}")
    
    def _create_basic_visualizations(self, df):
        """Create basic visualizations of experiment results"""
        # Figure 1: Pass Rate Comparison
        plt.figure(figsize=(15, 10))
        
        # Pass rate by model and approach
        plt.subplot(2, 2, 1)
        sns.barplot(x='model', y='pass_rate', hue='approach', data=df, errorbar=('ci', 95))
        plt.title('Pass Rate by Model and Approach', fontsize=14)
        plt.xlabel('Model')
        plt.ylabel('Pass Rate')
        plt.ylim(0, 1.05)
        
        # Complexity change
        plt.subplot(2, 2, 2)
        sns.boxplot(x='model', y='cc_diff', hue='approach', data=df)
        plt.title('Cyclomatic Complexity Change', fontsize=14)
        plt.xlabel('Model')
        plt.ylabel('Complexity Change')
        
        # Identifier entropy change
        plt.subplot(2, 2, 3)
        sns.boxplot(x='model', y='ie_diff', hue='approach', data=df)
        plt.title('Identifier Entropy Change', fontsize=14)
        plt.xlabel('Model')
        plt.ylabel('Entropy Change')
        
        # Obfuscation time
        plt.subplot(2, 2, 4)
        sns.boxplot(x='model', y='obfuscation_time', hue='approach', data=df)
        plt.title('Obfuscation Time (seconds)', fontsize=14)
        plt.xlabel('Model')
        plt.ylabel('Time (s)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config["output_dir"], "figures", "basic_metrics.png"), dpi=300)
        plt.close()
        
    def _create_advanced_visualizations(self, df):
        """Create more advanced visualizations of experiment results"""
        # Figure 2: Function category performance
        plt.figure(figsize=(15, 10))
        
        # Pass rate by category and model
        plt.subplot(2, 2, 1)
        category_order = sorted(df['category'].unique())
        sns.barplot(x='category', y='pass_rate', hue='model', data=df, 
                   order=category_order, errorbar=('ci', 95))
        plt.title('Pass Rate by Function Category and Model', fontsize=14)
        plt.xlabel('Function Category')
        plt.ylabel('Pass Rate')
        plt.ylim(0, 1.05)
        plt.xticks(rotation=45)
        plt.legend(title='Model')
        
        # Complexity change by category
        plt.subplot(2, 2, 2)
        sns.boxplot(x='category', y='cc_diff', hue='model', data=df, order=category_order)
        plt.title('Complexity Change by Category', fontsize=14)
        plt.xlabel('Function Category')
        plt.ylabel('Complexity Change')
        plt.xticks(rotation=45)
        plt.legend(title='Model')
        
        # Effect of approach by model
        plt.subplot(2, 2, 3)
        approach_effect = df.groupby(['model', 'function', 'approach'])['pass_rate'].mean().reset_index()
        approach_effect = approach_effect.pivot(index=['model', 'function'], columns='approach', values='pass_rate').reset_index()
        approach_effect['improvement'] = approach_effect['few-shot'] - approach_effect['standard']
        approach_effect_summary = approach_effect.groupby('model')['improvement'].mean().reset_index()
        
        sns.barplot(x='model', y='improvement', data=approach_effect_summary)
        plt.title('Average Improvement from Few-Shot Prompting', fontsize=14)
        plt.xlabel('Model')
        plt.ylabel('Pass Rate Improvement')
        plt.axhline(y=0, color='black', linestyle='--')
        
        # Correlation heatmap
        plt.subplot(2, 2, 4)
        corr_metrics = ['pass_rate', 'cc_diff', 'ie_diff', 'code_expansion', 
                       'semantic_elasticity', 'obfuscation_time']
        corr = df[corr_metrics].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title('Correlation Between Metrics', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config["output_dir"], "figures", "advanced_metrics.png"), dpi=300)
        plt.close()
        
    def _create_semantic_elasticity_visualizations(self, df):
        """Create visualizations focused on the Semantic Elasticity metric"""
        # Figure 3: Semantic Elasticity Analysis
        plt.figure(figsize=(15, 10))
        
        # Semantic Elasticity by model and approach
        plt.subplot(2, 2, 1)
        sns.boxplot(x='model', y='semantic_elasticity', hue='approach', data=df)
        plt.title('Semantic Elasticity by Model and Approach', fontsize=14)
        plt.xlabel('Model')
        plt.ylabel('Semantic Elasticity')
        
        # Semantic Elasticity by function category
        plt.subplot(2, 2, 2)
        category_order = sorted(df['category'].unique())
        sns.boxplot(x='category', y='semantic_elasticity', data=df, order=category_order)
        plt.title('Semantic Elasticity by Function Category', fontsize=14)
        plt.xlabel('Function Category')
        plt.ylabel('Semantic Elasticity')
        plt.xticks(rotation=45)
        
        # Semantic Elasticity vs. Pass Rate
        plt.subplot(2, 2, 3)
        sns.scatterplot(x='pass_rate', y='semantic_elasticity', hue='model', style='approach', data=df, alpha=0.7)
        plt.title('Semantic Elasticity vs. Pass Rate', fontsize=14)
        plt.xlabel('Pass Rate')
        plt.ylabel('Semantic Elasticity')
        
        # Components of Semantic Elasticity
        plt.subplot(2, 2, 4)
        
        # Filter to only successful cases
        success_df = df[df['pass_rate'] > 0.5]
        
        # Create scatter plot where:
        # - X-axis is complexity change
        # - Y-axis is code expansion
        # - Size is pass rate
        # - Color is model
        scatter = plt.scatter(
            x=success_df['cc_diff'], 
            y=success_df['code_expansion'],
            s=success_df['pass_rate'] * 100,  # Size based on pass rate
            c=success_df['semantic_elasticity'],  # Color based on SE score
            cmap='viridis',
            alpha=0.7
        )
        
        plt.colorbar(scatter, label='Semantic Elasticity')
        plt.title('Components of Semantic Elasticity', fontsize=14)
        plt.xlabel('Complexity Change')
        plt.ylabel('Code Expansion Ratio')
        plt.axvline(x=0, color='gray', linestyle='--')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config["output_dir"], "figures", "semantic_elasticity.png"), dpi=300)
        plt.close()
        
        # Figure 4: Model-specific Semantic Elasticity Analysis
        plt.figure(figsize=(15, 5 * len(df['model'].unique())))
        
        # For each model, create a scatter plot
        for i, model in enumerate(df['model'].unique()):
            model_data = df[df['model'] == model]
            
            plt.subplot(len(df['model'].unique()), 1, i+1)
            
            # Use approach as shape
            markers = {'standard': 'o', 'few-shot': '^'}
            
            for approach in model_data['approach'].unique():
                approach_data = model_data[model_data['approach'] == approach]
                
                # Create scatter plot with function as labels
                scatter = plt.scatter(
                    x=approach_data['cc_diff'],
                    y=approach_data['ie_diff'],
                    s=approach_data['pass_rate'] * 100,
                    c=approach_data['semantic_elasticity'],
                    cmap='viridis',
                    marker=markers[approach],
                    alpha=0.7,
                    label=approach
                )
            
            plt.colorbar(scatter, label='Semantic Elasticity')
            plt.title(f'{model}: Transformation Characteristics', fontsize=14)
            plt.xlabel('Complexity Change')
            plt.ylabel('Identifier Entropy Change')
            plt.axvline(x=0, color='gray', linestyle='--')
            plt.axhline(y=0, color='gray', linestyle='--')
            plt.legend(title='Approach')
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.config["output_dir"], "figures", "model_characteristics.png"), dpi=300)
        plt.close()


if __name__ == "__main__":
    # Check for command line arguments
    config_file = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Run experiments
    runner = ExperimentRunner(config_file)
    results = runner.run()
    
    print("Experiment complete!")