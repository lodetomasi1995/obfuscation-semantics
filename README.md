# Obfuscation Semantics

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A framework for evaluating LLM-driven code obfuscation using novel metrics like Semantic Elasticity.

## Overview

This repository contains the code and resources for the paper "Simplicity by Obfuscation: Evaluating LLM-Driven Code Transformation with Semantic Elasticity". It provides a comprehensive framework for:

1. Evaluating code obfuscation capabilities of different LLMs
2. Comparing standard and few-shot prompting approaches
3. Measuring obfuscation effectiveness using novel metrics
4. Benchmarking across diverse algorithmic patterns

## Features

- **Multi-Model Support**: Test Claude-3.5-Sonnet, Gemini-1.5, and GPT-4-Turbo
- **Diverse Algorithmic Patterns**: 30 functions across 5 categories
- **Comprehensive Metrics**: Includes pass rate, complexity, entropy, timing, and Semantic Elasticity
- **Configurable Experiments**: Easily customize which models, functions, and approaches to test
- **Detailed Analysis**: Generate visualizations and statistical comparisons

## Requirements

- Python 3.9+
- Required packages (install via `pip install -r requirements.txt`):
  - anthropic
  - google-generativeai
  - openai
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scipy

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/obfuscation-semantics.git
   cd obfuscation-semantics
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure API keys:
   - Create a copy of `config/example_config.json` as `config/config.json`
   - Add your API keys for Claude, Gemini, and GPT-4

## Usage

### Running Experiments

```bash
python src/experiment_runner.py --config config/config.json
```

You can also pass API keys directly:

```bash
python src/experiment_runner.py --anthropic-key YOUR_ANTHROPIC_KEY --gemini-key YOUR_GEMINI_KEY --openai-key YOUR_OPENAI_KEY
```

### Analyzing Results

```bash
python analysis/analysis.py --results-dir results/data
```

Generate visualizations:

```bash
python analysis/visualize.py --results-dir results/data
```

## Project Structure

```
obfuscation-semantics/
├── README.md
├── CITATION.cff
├── LICENSE
├── requirements.txt
├── config/
│   └── example_config.json
├── src/
│   ├── code_obfuscator.py
│   ├── semantic_elasticity.py
│   ├── experiment_runner.py
│   ├── test_cases.py
│   └── sample_functions.py
├── analysis/
│   ├── analysis.py
│   └── visualize.py
├── examples/
│   ├── standard_prompt_examples/
│   └── few_shot_examples/
└── results/
    ├── figures/
    └── data/
```

## Core Components

- **code_obfuscator.py**: Handles interaction with LLMs for code obfuscation
- **semantic_elasticity.py**: Implements the novel Semantic Elasticity metric
- **experiment_runner.py**: Coordinates execution of trials and collects results
- **test_cases.py**: Provides comprehensive test cases for function validation
- **sample_functions.py**: Contains the 30 functions used for benchmarking

## Semantic Elasticity Metric

Our Semantic Elasticity (SE) metric quantifies a model's ability to radically transform code structure while maintaining functionality:

SE = |ΔCC| × P² / E

Where:
- |ΔCC|: Absolute cyclomatic complexity change
- P²: Square of pass rate to emphasize functional correctness
- E: Code expansion ratio (inversely related)

Higher values indicate more effective transformations that maintain functionality while significantly changing code structure.

## Dataset

| Function Name | Category | Description | Algorithmic Pattern | Complexity | Uses Recursion | Parameters |
|---------------|----------|-------------|---------------------|------------|----------------|------------|
| factorial | Mathematical | Calculate factorial recursively | Recursive | O(n) | Yes | 1 |
| fibonacci | Mathematical | Calculate Fibonacci number | Recursive with overlapping subproblems | O(2ⁿ) | Yes | 1 |
| is_prime | Mathematical | Check if number is prime | Conditional logic | O(√n) | No | 1 |
| gcd | Mathematical | Find greatest common divisor | Euclidean algorithm | O(log(min(a,b))) | Yes | 2 |
| lcm | Mathematical | Find least common multiple | Mathematical calculation | O(log(min(a,b))) | Yes (via gcd) | 2 |
| power | Mathematical | Calculate power recursively | Recursive exponentiation | O(log n) | Yes | 2 |
| sqrt_newton | Mathematical | Calculate square root | Newton's method | O(log n) | No | 2 |
| bubble_sort | Sorting/Searching | Sort array using bubble sort | Nested iterations | O(n²) | No | 1 |
| binary_search | Sorting/Searching | Search in sorted array | Divide-and-conquer | O(log n) | No | 2 |
| merge_sort | Sorting/Searching | Sort using merge sort | Divide-and-conquer with recursion | O(n log n) | Yes | 1 |
| quick_sort | Sorting/Searching | Sort using quick sort | Partition-based sorting | O(n log n) avg, O(n²) worst | Yes | 1 |
| insertion_sort | Sorting/Searching | Sort using insertion | Iterative insertion | O(n²) | No | 1 |
| linear_search | Sorting/Searching | Search in unsorted array | Simple iteration | O(n) | No | 2 |
| str_reverse | String Manipulation | Reverse a string | Simple string manipulation | O(n) | No | 1 |
| is_palindrome | String Manipulation | Check if string is palindrome | String testing | O(n) | No | 1 |
| word_count | String Manipulation | Count words in text | Basic text processing | O(n) | No | 1 |
| longest_common_substring | String Manipulation | Find common substring | Dynamic programming | O(m×n) | No | 2 |
| levenshtein_distance | String Manipulation | Calculate edit distance | Edit distance algorithm | O(m×n) | No | 2 |
| count_vowels | String Manipulation | Count vowels in string | Character filtering | O(n) | No | 1 |
| flatten_list | Data Structure | Flatten nested list | Recursive list transformation | O(n) | Yes | 1 |
| list_permutations | Data Structure | Generate all permutations | Combinatorial algorithm | O(n!) | Yes | 1 |
| dict_merge | Data Structure | Merge dictionaries recursively | Nested structure merging | O(n+m) | Yes | 2 |
| remove_duplicates | Data Structure | Remove duplicates from list | Set operations | O(n) | No | 1 |
| rotate_array | Data Structure | Rotate array elements | Array manipulation | O(n) | No | 2 |
| tower_of_hanoi | Recursive | Solve Tower of Hanoi puzzle | Classic recursion problem | O(2ⁿ) | Yes | 1 |
| binary_tree_depth | Recursive | Find max depth of binary tree | Tree traversal | O(n) | Yes | 1 |
| flood_fill | Recursive | Perform flood fill on image | Graph traversal | O(n) | Yes | 4 |
| knapsack | Recursive | Solve knapsack problem | Optimization problem | O(n×W) | Yes | 3 |
| edit_distance | Recursive | Calculate edit distance | String comparison | O(m×n) | No | 2 |
| coin_change | Recursive | Find minimum coins for amount | Dynamic programming | O(n×amount) | No | 2 |
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

