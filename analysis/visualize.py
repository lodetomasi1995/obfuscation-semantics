"""
Visualization module for code obfuscation experiment results.

This module creates various visualizations to help analyze and understand
the results of code obfuscation experiments across different models,
approaches, and function categories.
"""

import os
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
sns.set_context("paper", font_scale=1.2)


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


def plot_pass_rate_comparison(df: pd.DataFrame, output_dir: str):
    """
    Create pass rate comparison visualizations.
    
    Args:
        df: DataFrame with experiment results
        output_dir: Directory to save plots
    """
    plt.figure(figsize=(16, 12))

    # 1a. Overall pass rate by approach
    plt.subplot(2, 2, 1)
    sns.barplot(x='approach', y='pass_rate', data=df, errorbar=('ci', 95))
    plt.title('Pass Rate by Approach', fontsize=14)
    plt.xlabel('Approach')
    plt.ylabel('Pass Rate')
    plt.ylim(0, 1.05)

    # 1b. Pass rate by model and approach
    plt.subplot(2, 2, 2)
    sns.barplot(x='model', y='pass_rate', hue='approach', data=df, errorbar=('ci', 95))
    plt.title('Pass Rate by Model and Approach', fontsize=14)
    plt.xlabel('Model')
    plt.ylabel('Pass Rate')
    plt.ylim(0, 1.05)
    plt.legend(title='Approach')

    # 1c. Pass rate by function category and approach
    plt.subplot(2, 2, 3)
    category_order = ["Mathematical", "Sorting & Searching", "String Manipulation", "Data Structure", "Recursive"]
    sns.barplot(x='category', y='pass_rate', hue='approach', data=df, order=category_order, errorbar=('ci', 95))
    plt.title('Pass Rate by Function Category and Approach', fontsize=14)
    plt.xlabel('Function Category')
    plt.ylabel('Pass Rate')
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45)
    plt.legend(title='Approach')

    # 1d. Pass rate for specific models
    plt.subplot(2, 2, 4)
    model_pass_rate = df.groupby(['model', 'approach'])['pass_rate'].mean().reset_index()
    model_pass_rate = model_pass_rate.pivot(index='model', columns='approach', values='pass_rate')
    model_pass_rate.plot(kind='bar', ax=plt.gca())
    plt.title('Model-Specific Pass Rates by Approach', fontsize=14)
    plt.xlabel('Model')
    plt.ylabel('Pass Rate')
    plt.ylim(0, 1.05)
    plt.legend(title='Approach')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pass_rate_comparisons.png'), dpi=300)
    plt.close()


def plot_complexity_metrics(df: pd.DataFrame, output_dir: str):
    """
    Create complexity metrics visualizations.
    
    Args:
        df: DataFrame with experiment results
        output_dir: Directory to save plots
    """
    plt.figure(figsize=(16, 12))

    # 2a. Cyclomatic complexity change by approach
    plt.subplot(2, 2, 1)
    sns.boxplot(x='approach', y='cc_diff', data=df)
    plt.title('Cyclomatic Complexity Change by Approach', fontsize=14)
    plt.xlabel('Approach')
    plt.ylabel('Complexity Change')

    # 2b. Cyclomatic complexity change by model and approach
    plt.subplot(2, 2, 2)
    sns.boxplot(x='model', y='cc_diff', hue='approach', data=df)
    plt.title('Complexity Change by Model and Approach', fontsize=14)
    plt.xlabel('Model')
    plt.ylabel('Complexity Change')
    plt.legend(title='Approach')

    # 2c. Identifier entropy change by approach
    plt.subplot(2, 2, 3)
    sns.boxplot(x='approach', y='ie_diff', data=df)
    plt.title('Identifier Entropy Change by Approach', fontsize=14)
    plt.xlabel('Approach')
    plt.ylabel('Entropy Change')

    # 2d. Identifier entropy change by model and approach
    plt.subplot(2, 2, 4)
    sns.boxplot(x='model', y='ie_diff', hue='approach', data=df)
    plt.title('Entropy Change by Model and Approach', fontsize=14)
    plt.xlabel('Model')
    plt.ylabel('Entropy Change')
    plt.legend(title='Approach')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'complexity_metrics_comparisons.png'), dpi=300)
    plt.close()


def plot_semantic_elasticity(df: pd.DataFrame, output_dir: str):
    """
    Create semantic elasticity visualizations.
    
    Args:
        df: DataFrame with experiment results
        output_dir: Directory to save plots
    """
    plt.figure(figsize=(16, 12))

    # 3a. Semantic elasticity by approach
    plt.subplot(2, 2, 1)
    sns.boxplot(x='approach', y='semantic_elasticity', data=df)
    plt.title('Semantic Elasticity by Approach', fontsize=14)
    plt.xlabel('Approach')
    plt.ylabel('Semantic Elasticity')

    # 3b. Semantic elasticity by model and approach
    plt.subplot(2, 2, 2)
    sns.boxplot(x='model', y='semantic_elasticity', hue='approach', data=df)
    plt.title('Semantic Elasticity by Model and Approach', fontsize=14)
    plt.xlabel('Model')
    plt.ylabel('Semantic Elasticity')
    plt.legend(title='Approach')

    # 3c. Semantic elasticity by function category
    plt.subplot(2, 2, 3)
    category_order = ["Mathematical", "Sorting & Searching", "String Manipulation", "Data Structure", "Recursive"]
    sns.boxplot(x='category', y='semantic_elasticity', data=df, order=category_order)
    plt.title('Semantic Elasticity by Function Category', fontsize=14)
    plt.xlabel('Function Category')
    plt.ylabel('Semantic Elasticity')
    plt.xticks(rotation=45)

    # 3d. Top semantic elasticity values
    plt.subplot(2, 2, 4)
    # Group by model and approach, take mean and max
    se_summary = df.groupby(['model', 'approach']).agg({
        'semantic_elasticity': ['mean', 'max']
    }).reset_index()
    
    # Flatten MultiIndex columns
    se_summary.columns = ['_'.join(col).strip('_') for col in se_summary.columns.values]
    
    # Sort by mean
    se_summary = se_summary.sort_values('semantic_elasticity_mean', ascending=False)
    
    # Plot a grouped bar chart
    x = range(len(se_summary))
    width = 0.35
    
    fig, ax = plt.subplot(2, 2, 4)
    ax.bar(x, se_summary['semantic_elasticity_mean'], width, label='Mean', color='skyblue')
    ax.bar([i + width for i in x], se_summary['semantic_elasticity_max'], width, label='Max', color='salmon')
    
    # Add labels and title
    ax.set_ylabel('Semantic Elasticity')
    ax.set_title('Top Semantic Elasticity Values by Model/Approach', fontsize=14)
    ax.set_xticks([i + width/2 for i in x])
    ax.set_xticklabels([f"{row['model']}\n({row['approach']})" for _, row in se_summary.iterrows()])
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'semantic_elasticity_comparisons.png'), dpi=300)
    plt.close()


def plot_correlation_matrix(df: pd.DataFrame, output_dir: str):
    """
    Create correlation matrix visualization.
    
    Args:
        df: DataFrame with experiment results
        output_dir: Directory to save plots
    """
    # Select numeric columns for correlation
    numeric_cols = ['pass_rate', 'code_expansion', 'cc_diff', 'ie_diff', 
                  'semantic_elasticity', 'avg_timing_diff_ms']
    
    # Drop rows with missing values
    df_clean = df[numeric_cols].dropna()
    
    # Calculate correlation matrix
    corr = df_clean.corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
    plt.title('Correlation Matrix of Obfuscation Metrics', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=300)
    plt.close()


def plot_model_performance_radar(df: pd.DataFrame, output_dir: str):
    """
    Create radar chart comparing model performance.
    
    Args:
        df: DataFrame with experiment results
        output_dir: Directory to save plots
    """
    # Calculate mean metrics by model
    model_metrics = df.groupby('model').agg({
        'pass_rate': 'mean',
        'code_expansion': 'mean',
        'ie_diff': 'mean',
        'semantic_elasticity': 'mean',
        'obfuscation_time': 'mean'
    }).reset_index()
    
    # Normalize each metric between 0 and 1 for radar chart
    for col in model_metrics.columns:
        if col != 'model':
            # For obfuscation_time, lower is better
            if col == 'obfuscation_time':
                model_metrics[col] = 1 - (model_metrics[col] - model_metrics[col].min()) / (model_metrics[col].max() - model_metrics[col].min())
            else:
                model_metrics[col] = (model_metrics[col] - model_metrics[col].min()) / (model_metrics[col].max() - model_metrics[col].min())
    
    # Set up the radar chart
    categories = ['Pass Rate', 'Code Expansion', 'Identifier Entropy', 'Semantic Elasticity', 'Processing Speed']
    N = len(categories)
    
    # Create angles for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create figure
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    
    # Set category labels
    plt.xticks(angles[:-1], categories, size=12)
    
    # Remove radial labels
    plt.yticks([])
    
    # Draw y-axis gridlines
    ax.set_rlabel_position(0)
    plt.ylim(0, 1)
    
    # Plot each model
    for i, row in model_metrics.iterrows():
        model = row['model']
        values = [row['pass_rate'], row['code_expansion'], row['ie_diff'], 
                 row['semantic_elasticity'], 1 - row['obfuscation_time']]
        values += values[:1]  # Close the loop
        
        # Plot values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
        ax.fill(angles, values, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title('Model Performance Comparison', size=20, y=1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_performance_radar.png'), dpi=300)
    plt.close()


def plot_paradoxical_simplification(df: pd.DataFrame, output_dir: str):
    """
    Visualize the paradoxical code simplification phenomenon.
    
    Args:
        df: DataFrame with experiment results
        output_dir: Directory to save plots
    """
    plt.figure(figsize=(16, 8))
    
    # Left panel: Complexity change by approach
    plt.subplot(1, 2, 1)
    sns.boxplot(x='approach', y='cc_diff', data=df)
    plt.axhline(0, color='gray', linestyle='--')
    plt.title('Complexity Change by Approach', fontsize=14)
    plt.xlabel('Approach')
    plt.ylabel('Cyclomatic Complexity Change')
    
    # Shade region where complexity increases (traditional expectation)
    ax = plt.gca()
    ylim = ax.get_ylim()
    plt.fill_between([ax.get_xlim()[0], ax.get_xlim()[1]], 0, ylim[1], alpha=0.1, color='gray')
    plt.annotate('Traditional Expectation\n(Complexity Increase)', 
                xy=(0.5, ylim[1]*0.75), ha='center', fontsize=10)
    
    # Right panel: Scatter plot of pass rate vs complexity change
    plt.subplot(1, 2, 2)
    sns.scatterplot(x='cc_diff', y='pass_rate', hue='model', data=df, alpha=0.7)
    plt.axvline(0, color='gray', linestyle='--')
    plt.title('Pass Rate vs. Complexity Change', fontsize=14)
    plt.xlabel('Cyclomatic Complexity Change')
    plt.ylabel('Pass Rate')
    
    # Calculate correlation for annotation
    corr = df['cc_diff'].corr(df['pass_rate'])
    plt.annotate(f'Correlation: {corr:.2f}', xy=(0.05, 0.95), 
                xycoords='axes fraction', ha='left', va='top', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'paradoxical_simplification.png'), dpi=300)
    plt.close()


def plot_function_heatmaps(df: pd.DataFrame, output_dir: str):
    """
    Create heatmaps showing performance across functions.
    
    Args:
        df: DataFrame with experiment results
        output_dir: Directory to save plots
    """
    # Create heatmaps for standard and few-shot approaches
    plt.figure(figsize=(20, 10))
    
    # Filter by approach
    standard_df = df[df['approach'] == 'standard']
    fewshot_df = df[df['approach'] == 'few-shot']
    
    # 1. Standard approach heatmap
    plt.subplot(1, 2, 1)
    heatmap_data_std = standard_df.pivot_table(
        values='pass_rate',
        index='function',
        columns='model',
        aggfunc='mean'
    )
    sns.heatmap(heatmap_data_std, annot=True, cmap='YlGnBu', fmt='.2f', vmin=0, vmax=1)
    plt.title('Standard Prompting Pass Rate by Function and Model', fontsize=14)
    
    # 2. Few-shot approach heatmap
    plt.subplot(1, 2, 2)
    heatmap_data_fs = fewshot_df.pivot_table(
        values='pass_rate',
        index='function',
        columns='model',
        aggfunc='mean'
    )
    sns.heatmap(heatmap_data_fs, annot=True, cmap='YlGnBu', fmt='.2f', vmin=0, vmax=1)
    plt.title('Few-Shot Prompting Pass Rate by Function and Model', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'function_heatmaps.png'), dpi=300)
    plt.close()


def plot_semantic_elasticity_distribution(df: pd.DataFrame, output_dir: str):
    """
    Plot distribution of semantic elasticity values.
    
    Args:
        df: DataFrame with experiment results
        output_dir: Directory to save plots
    """
    plt.figure(figsize=(12, 8))
    
    # Kernel density estimation plot by model
    sns.kdeplot(data=df, x='semantic_elasticity', hue='model', fill=True, alpha=0.3)
    
    # Add vertical lines for mean values
    for model in df['model'].unique():
        mean_value = df[df['model'] == model]['semantic_elasticity'].mean()
        plt.axvline(mean_value, linestyle='--', 
                   color=sns.color_palette()[list(df['model'].unique()).index(model)],
                   label=f'{model} mean ({mean_value:.3f})')
    
    plt.title('Distribution of Semantic Elasticity by Model', fontsize=16)
    plt.xlabel('Semantic Elasticity')
    plt.ylabel('Density')
    plt.legend(title='Model')
    
    # Add annotation explaining the metric
    plt.annotate('Higher Semantic Elasticity = More radical transformation\nwhile maintaining functionality',
                xy=(0.98, 0.98), xycoords='axes fraction', ha='right', va='top',
                bbox=dict(boxstyle='round', fc='white', alpha=0.8), fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'semantic_elasticity_distribution.png'), dpi=300)
    plt.close()


def generate_all_visualizations(results_path: str, output_dir: str):
    """
    Generate all visualizations from experiment results.
    
    Args:
        results_path: Path to results file
        output_dir: Directory to save visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and prepare data
    print(f"Loading results from {results_path}...")
    df = load_results(results_path)
    df = categorize_functions(df)
    print(f"Loaded {len(df)} experiment results")
    
    # Generate visualizations
    print("Generating pass rate comparison visualizations...")
    plot_pass_rate_comparison(df, output_dir)
    
    print("Generating complexity metrics visualizations...")
    plot_complexity_metrics(df, output_dir)
    
    print("Generating semantic elasticity visualizations...")
    plot_semantic_elasticity(df, output_dir)
    
    print("Generating correlation matrix...")
    plot_correlation_matrix(df, output_dir)
    
    print("Generating model performance radar chart...")
    plot_model_performance_radar(df, output_dir)
    
    print("Generating paradoxical simplification visualization...")
    plot_paradoxical_simplification(df, output_dir)
    
    print("Generating function heatmaps...")
    plot_function_heatmaps(df, output_dir)
    
    print("Generating semantic elasticity distribution plot...")
    plot_semantic_elasticity_distribution(df, output_dir)
    
    print(f"All visualizations saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate visualizations for obfuscation results")
    parser.add_argument("--results", help="Path to results CSV or JSON file", required=True)
    parser.add_argument("--output", help="Directory for saving visualizations", default="results/figures")
    args = parser.parse_args()
    
    generate_all_visualizations(args.results, args.output)