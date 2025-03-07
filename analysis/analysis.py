"""
Statistical Analysis of Code Obfuscation Results.

This module processes experiment results to extract insights about the 
performance of different models and approaches for code obfuscation.
"""

import os
import argparse
import json
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Any, Tuple


def load_results(results_path: str) -> pd.DataFrame:
    """
    Load results from CSV or JSON file.
    
    Args:
        results_path: Path to results file
        
    Returns:
        DataFrame containing results
    """
    if results_path.endswith('.csv'):
        df = pd.read_csv(results_path)
    elif results_path.endswith('.json'):
        df = pd.read_json(results_path)
    else:
        raise ValueError(f"Unsupported file format: {results_path}")
    
    # Convert JSON strings back to dictionaries if needed
    for col in ['cyclomatic_complexity', 'identifier_entropy']:
        if col in df.columns and isinstance(df[col].iloc[0], str):
            df[col] = df[col].apply(json.loads)
    
    return df


def categorize_functions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add function category based on function name.
    
    Args:
        df: DataFrame with function data
        
    Returns:
        DataFrame with added category column
    """
    mathematical_functions = [
        "factorial", "fibonacci", "is_prime", "gcd", "lcm", 
        "power", "sqrt_newton"
    ]
    
    sorting_searching = [
        "bubble_sort", "binary_search", "merge_sort", "quick_sort", 
        "insertion_sort", "linear_search"
    ]
    
    string_functions = [
        "str_reverse", "is_palindrome", "word_count", 
        "longest_common_substring", "levenshtein_distance", "count_vowels"
    ]
    
    data_structure = [
        "flatten_list", "list_permutations", "dict_merge", 
        "remove_duplicates", "rotate_array"
    ]
    
    recursive = [
        "tower_of_hanoi", "binary_tree_depth", "flood_fill", 
        "knapsack", "edit_distance", "coin_change"
    ]
    
    # Create category mapping
    category_map = {}
    for func in mathematical_functions:
        category_map[func] = "Mathematical"
    for func in sorting_searching:
        category_map[func] = "Sorting & Searching"
    for func in string_functions:
        category_map[func] = "String Manipulation" 
    for func in data_structure:
        category_map[func] = "Data Structure"
    for func in recursive:
        category_map[func] = "Recursive"
    
    # Add category column
    df['category'] = df['function'].map(category_map)
    
    return df


def calculate_summary_metrics(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Calculate summary metrics for different groupings.
    
    Args:
        df: DataFrame with experiment results
        
    Returns:
        Dictionary of DataFrames with summary metrics
    """
    # 1. Overall metrics by approach
    overall_approach = df.groupby('approach').agg({
        'pass_rate': ['mean', 'std'],
        'code_expansion': ['mean', 'std'],
        'cc_diff': ['mean', 'std'],
        'ie_diff': ['mean', 'std'],
        'obfuscation_time': ['mean', 'std'],
        'semantic_elasticity': ['mean', 'std']
    }).reset_index()
    
    # 2. Metrics by model and approach
    model_approach = df.groupby(['model', 'approach']).agg({
        'pass_rate': ['mean', 'std'],
        'code_expansion': ['mean', 'std'],
        'cc_diff': ['mean', 'std'],
        'ie_diff': ['mean', 'std'],
        'obfuscation_time': ['mean', 'std'],
        'semantic_elasticity': ['mean', 'std']
    }).reset_index()
    
    # 3. Metrics by category and approach
    category_approach = df.groupby(['category', 'approach']).agg({
        'pass_rate': ['mean', 'std'],
        'code_expansion': ['mean', 'std'],
        'cc_diff': ['mean', 'std'],
        'ie_diff': ['mean', 'std'],
        'obfuscation_time': ['mean', 'std'],
        'semantic_elasticity': ['mean', 'std']
    }).reset_index()
    
    # 4. Metrics by function, model, and approach
    function_model_approach = df.groupby(['function', 'model', 'approach']).agg({
        'pass_rate': 'mean',
        'code_expansion': 'mean',
        'cc_diff': 'mean',
        'ie_diff': 'mean',
        'obfuscation_time': 'mean',
        'semantic_elasticity': 'mean'
    }).reset_index()
    
    return {
        'overall_approach': overall_approach,
        'model_approach': model_approach,
        'category_approach': category_approach,
        'function_model_approach': function_model_approach
    }


def run_statistical_tests(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Run statistical tests to compare approaches.
    
    Args:
        df: DataFrame with experiment results
        
    Returns:
        Dictionary with test results
    """
    results = {}
    
    # Split data by approach
    standard_df = df[df['approach'] == 'standard']
    fewshot_df = df[df['approach'] == 'few-shot']
    
    # T-test on pass rates
    t_stat, p_value = stats.ttest_ind(
        standard_df['pass_rate'], 
        fewshot_df['pass_rate'],
        equal_var=False  # Welch's t-test (doesn't assume equal variance)
    )
    results['pass_rate_ttest'] = {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'mean_standard': standard_df['pass_rate'].mean(),
        'mean_fewshot': fewshot_df['pass_rate'].mean()
    }
    
    # T-test on complexity change
    t_stat, p_value = stats.ttest_ind(
        standard_df['cc_diff'], 
        fewshot_df['cc_diff'],
        equal_var=False
    )
    results['cc_diff_ttest'] = {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'mean_standard': standard_df['cc_diff'].mean(),
        'mean_fewshot': fewshot_df['cc_diff'].mean()
    }
    
    # T-test on entropy change
    t_stat, p_value = stats.ttest_ind(
        standard_df['ie_diff'], 
        fewshot_df['ie_diff'],
        equal_var=False
    )
    results['ie_diff_ttest'] = {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'mean_standard': standard_df['ie_diff'].mean(),
        'mean_fewshot': fewshot_df['ie_diff'].mean()
    }
    
    # T-test on semantic elasticity
    t_stat, p_value = stats.ttest_ind(
        standard_df['semantic_elasticity'], 
        fewshot_df['semantic_elasticity'],
        equal_var=False
    )
    results['semantic_elasticity_ttest'] = {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'mean_standard': standard_df['semantic_elasticity'].mean(),
        'mean_fewshot': fewshot_df['semantic_elasticity'].mean()
    }
    
    # Correlation analysis
    correlations = {}
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        
        # Correlation between pass rate and complexity change
        corr, p_value = stats.pearsonr(model_df['pass_rate'], model_df['cc_diff'])
        correlations[f'{model}_pass_rate_cc_diff'] = {
            'correlation': corr,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        # Correlation between pass rate and semantic elasticity
        corr, p_value = stats.pearsonr(model_df['pass_rate'], model_df['semantic_elasticity'])
        correlations[f'{model}_pass_rate_semantic_elasticity'] = {
            'correlation': corr,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    results['correlations'] = correlations
    
    # ANOVA for model comparison
    models = df['model'].unique()
    if len(models) >= 3:  # Need at least 3 groups for ANOVA
        model_groups = [df[df['model'] == model]['pass_rate'] for model in models]
        f_stat, p_value = stats.f_oneway(*model_groups)
        results['model_anova'] = {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'models': list(models)
        }
    
    return results


def calculate_semantic_elasticity_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank models and approaches by semantic elasticity.
    
    Args:
        df: DataFrame with experiment results
        
    Returns:
        DataFrame with rankings
    """
    # Group by model and approach
    se_rankings = df.groupby(['model', 'approach']).agg({
        'semantic_elasticity': ['mean', 'std', 'max'],
        'pass_rate': 'mean'
    }).reset_index()
    
    # Flatten MultiIndex columns
    se_rankings.columns = ['_'.join(col).strip('_') for col in se_rankings.columns.values]
    
    # Sort by mean semantic elasticity
    se_rankings = se_rankings.sort_values('semantic_elasticity_mean', ascending=False)
    
    # Add rank column
    se_rankings['rank'] = range(1, len(se_rankings) + 1)
    
    return se_rankings


def analyze_results(results_path: str, output_dir: str = None):
    """
    Perform comprehensive analysis of experiment results.
    
    Args:
        results_path: Path to results file
        output_dir: Directory for saving output files
    """
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    print(f"Loading results from {results_path}...")
    df = load_results(results_path)
    print(f"Loaded {len(df)} experiment results")
    
    # Add function categories
    df = categorize_functions(df)
    
    # Calculate summary metrics
    print("Calculating summary metrics...")
    summary_metrics = calculate_summary_metrics(df)
    
    # Run statistical tests
    print("Running statistical tests...")
    stats_results = run_statistical_tests(df)
    
    # Calculate semantic elasticity rankings
    print("Calculating semantic elasticity rankings...")
    se_rankings = calculate_semantic_elasticity_rankings(df)
    
    # Print key findings
    print("\n" + "="*50)
    print("KEY FINDINGS")
    print("="*50)
    
    # 1. Overall approach comparison
    overall = summary_metrics['overall_approach']
    print("\nOverall Performance by Approach:")
    for _, row in overall.iterrows():
        approach = row['approach']
        pass_rate = row[('pass_rate', 'mean')] * 100
        pass_rate_std = row[('pass_rate', 'std')] * 100
        cc_diff = row[('cc_diff', 'mean')]
        ie_diff = row[('ie_diff', 'mean')]
        se = row[('semantic_elasticity', 'mean')]
        
        print(f"  {approach.upper()}:")
        print(f"   - Pass Rate: {pass_rate:.1f}% (±{pass_rate_std:.1f}%)")
        print(f"   - Complexity Change: {cc_diff:.2f}")
        print(f"   - Identifier Entropy Change: {ie_diff:.2f}")
        print(f"   - Semantic Elasticity: {se:.3f}")
    
    # 2. Model-specific findings
    print("\nModel Performance:")
    model_approach = summary_metrics['model_approach']
    for model in df['model'].unique():
        model_data = model_approach[model_approach['model'] == model]
        
        print(f"  {model.upper()}:")
        for _, row in model_data.iterrows():
            approach = row['approach']
            pass_rate = row[('pass_rate', 'mean')] * 100
            se = row[('semantic_elasticity', 'mean')]
            
            print(f"   - {approach}: {pass_rate:.1f}% pass rate, {se:.3f} semantic elasticity")
    
    # 3. Statistical significance
    print("\nStatistical Significance:")
    for test_name, test_results in stats_results.items():
        if test_name == 'correlations' or test_name == 'model_anova':
            continue
            
        print(f"  {test_name}:")
        print(f"   - p-value: {test_results['p_value']:.4f}")
        print(f"   - Significant: {test_results['significant']}")
        print(f"   - Mean (standard): {test_results['mean_standard']:.4f}")
        print(f"   - Mean (few-shot): {test_results['mean_fewshot']:.4f}")
    
    # 4. Top performers
    print("\nTop Performers by Semantic Elasticity:")
    for i, (_, row) in enumerate(se_rankings.iterrows()):
        if i >= 3:  # Show only top 3
            break
            
        model = row['model']
        approach = row['approach']
        se_mean = row['semantic_elasticity_mean']
        se_max = row['semantic_elasticity_max']
        pass_rate = row['pass_rate_mean'] * 100
        
        print(f"  {i+1}. {model} ({approach}):")
        print(f"     - Mean SE: {se_mean:.3f}")
        print(f"     - Max SE: {se_max:.3f}")
        print(f"     - Pass Rate: {pass_rate:.1f}%")
    
    # Save results if output directory specified
    if output_dir:
        # Save summary metrics
        for name, data in summary_metrics.items():
            data.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)
        
        # Save semantic elasticity rankings
        se_rankings.to_csv(os.path.join(output_dir, "semantic_elasticity_rankings.csv"), index=False)
        
        # Save statistical test results
        with open(os.path.join(output_dir, "statistical_tests.json"), 'w') as f:
            # Convert numpy values to Python native types for JSON serialization
            json_stats = {}
            for key, value in stats_results.items():
                if isinstance(value, dict):
                    json_stats[key] = {k: float(v) if isinstance(v, np.number) else v 
                                      for k, v in value.items()}
                else:
                    json_stats[key] = float(value) if isinstance(value, np.number) else value
            
            json.dump(json_stats, f, indent=2)
        
        print(f"\nAnalysis results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze code obfuscation results")
    parser.add_argument("--results", help="Path to results CSV or JSON file", required=True)
    parser.add_argument("--output", help="Directory for saving output files")
    args = parser.parse_args()
    
    analyze_results(args.results, args.output)