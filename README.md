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



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

