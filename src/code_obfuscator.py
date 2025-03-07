"""
LLM-Driven Code Obfuscation with Semantic Elasticity
===================================================
Main implementation of the CodeObfuscator class for evaluating LLM capabilities
in code obfuscation while preserving functionality.

Author: Lorenzo De Tomasi
Institution: Università degli Studi dell'Aquila
"""

import ast
import random
import time
import math
import traceback
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional, Union
import anthropic
import google.generativeai as genai
from openai import OpenAI

class DeadCodeGenerator:
    """Generates sophisticated dead code for multi-layered obfuscation"""

    @staticmethod
    def get_random_deadcode() -> str:
        """Return a randomly selected dead code pattern for insertion"""
        patterns = [
            # Basic unused variables and operations
            "# Unused variable\n_ø = 1",
            "# Canceled operation\n_δ = 1 - 1",
            "# Always true statement\nif 'x'[0] == 'x': pass",
            
            # More sophisticated patterns
            "# Complex mathematical identity that evaluates to zero\n_ξ = int(math.sin(math.pi)**2 * 10**10)",
            "# Logical contradiction wrapped in a conditional\nif False and '_' != '_': raise ValueError('Unreachable')",
            
            # Unused function definitions
            "# Empty function\ndef _φ(*args, **kwargs): return args[0] if args else None",
            "# Recursive function that's never called\ndef _ψ(n): return 1 if n <= 0 else n * _ψ(n-1) / n",
            
            # Deceptive operations
            "# XOR with zero\n_μ = 0b10101010 ^ 0 ^ 0",
            "# Double negation\n_β = not not True",
            "# Lambda function that does nothing\n_λ = lambda *x: x[0] if x else None",
            
            # Class definitions
            """# Unused class that looks important
class _Validator:
    def __init__(self):
        self.is_valid = True
    def validate(self, value):
        return isinstance(value, (int, float))""",
            
            # Operations that appear to have side effects
            """# Context manager that does nothing
class _NoOp:
    def __enter__(self): return self
    def __exit__(self, *args): pass
_ctx = _NoOp()""",
            
            # Opaque predicates
            """# Opaque predicate (always False)
def _is_perfect_square(n):
    return n > 0 and int(math.sqrt(n)) ** 2 == n
_check = _is_perfect_square(2)"""
        ]
        return random.choice(patterns)


class CodeObfuscator:
    """Main class for LLM-driven code obfuscation evaluation"""

    def __init__(self, anthropic_api_key=None, gemini_api_key=None, openai_api_key=None):
        """Initialize clients for different LLMs"""
        self.dead_code_gen = DeadCodeGenerator()
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key) if anthropic_api_key else None
        self.gemini_api_key = gemini_api_key
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-pro-latest')
        else:
            self.gemini_model = None
        self.openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None

    def compute_cyclomatic_complexity(self, code: str) -> int:
        """Calculate McCabe's cyclomatic complexity using AST analysis"""
        class ComplexityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.complexity = 1  # Base complexity is 1

            def visit_If(self, node):
                self.complexity += 1
                self.generic_visit(node)

            def visit_While(self, node):
                self.complexity += 1
                self.generic_visit(node)

            def visit_For(self, node):
                self.complexity += 1
                self.generic_visit(node)

            def visit_Lambda(self, node):
                self.complexity += 1
                self.generic_visit(node)
                
            def visit_BoolOp(self, node):
                # Count boolean operations (and, or) as decision points
                if isinstance(node.op, (ast.And, ast.Or)):
                    self.complexity += len(node.values) - 1
                self.generic_visit(node)

            def visit_Try(self, node):
                # Count each except handler as a branch
                self.complexity += len(node.handlers)
                self.generic_visit(node)

        try:
            tree = ast.parse(code)
            visitor = ComplexityVisitor()
            visitor.visit(tree)
            return visitor.complexity
        except:
            return 0

    def compute_identifier_entropy(self, code: str) -> float:
        """Calculate Shannon entropy of identifiers to measure naming obfuscation quality"""
        try:
            tree = ast.parse(code)
            identifiers = []

            class IdentifierVisitor(ast.NodeVisitor):
                def visit_Name(self, node):
                    identifiers.append(node.id)
                    self.generic_visit(node)
                    
                def visit_FunctionDef(self, node):
                    identifiers.append(node.name)
                    self.generic_visit(node)
                    
                def visit_ClassDef(self, node):
                    identifiers.append(node.name)
                    self.generic_visit(node)
                    
                def visit_arg(self, node):
                    if node.arg:
                        identifiers.append(node.arg)
                    self.generic_visit(node)

            visitor = IdentifierVisitor()
            visitor.visit(tree)

            if not identifiers:
                return 0

            freqs = Counter(identifiers)
            total = sum(freqs.values())
            probs = [count/total for count in freqs.values()]
            return -sum(p * math.log2(p) for p in probs)
        except Exception as e:
            print(f"Error computing identifier entropy: {e}")
            return 0

    def obfuscate_code(self, python_code: str, model_name: str = "gemini") -> str:
        """Obfuscate Python code while maintaining functionality using standard prompting"""
        # Validate model availability
        self._validate_model_availability(model_name)
        
        # Extract original function name
        tree = ast.parse(python_code)
        orig_func_name = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                orig_func_name = node.name
                break
                
        if not orig_func_name:
            raise ValueError("No function definition found in the provided code")

        # Generate new obfuscated name
        new_func_name = self._generate_obfuscated_name()
        
        # Generate prompt for the specified model
        prompt = self._create_standard_prompt(orig_func_name, new_func_name)
        
        # Generate obfuscated code using the appropriate model
        obfuscated = self._generate_with_model(prompt, model_name)
        
        # Clean and validate the generated code
        return self._clean_and_validate_code(obfuscated)

    def obfuscate_code_with_examples(self, python_code: str, model_name: str = "gemini") -> str:
        """Obfuscate Python code using few-shot learning with examples"""
        # Validate model availability
        self._validate_model_availability(model_name)
        
        # Extract original function name
        tree = ast.parse(python_code)
        orig_func_name = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                orig_func_name = node.name
                break
                
        if not orig_func_name:
            raise ValueError("No function definition found in the provided code")

        # Generate new obfuscated name
        new_func_name = self._generate_obfuscated_name()
        
        # Generate example-based prompt
        prompt = self._create_few_shot_prompt(python_code, orig_func_name, new_func_name)
        
        # Generate obfuscated code using the appropriate model
        obfuscated = self._generate_with_model(prompt, model_name)
        
        # Clean and validate the generated code
        return self._clean_and_validate_code(obfuscated)

    def _generate_obfuscated_name(self) -> str:
        """Generate an obfuscated name using confusable characters"""
        # Extended character set including confusable characters and Unicode homoglyphs
        chars = "IOl01xX_ƟØµ"
        length = random.randint(3, 7)
        return '_' + ''.join(random.choice(chars) for _ in range(length))
    
    def _validate_model_availability(self, model_name: str):
        """Validate that the requested model is available"""
        if model_name == "claude" and self.anthropic_client is None:
            raise ValueError("Anthropic API key not provided. Cannot use Claude model.")
        elif model_name == "gemini" and self.gemini_model is None:
            raise ValueError("Gemini API key not provided. Cannot use Gemini model.")
        elif model_name == "gpt-4" and self.openai_client is None:
            raise ValueError("OpenAI API key not provided. Cannot use GPT-4 model.")

    def _create_standard_prompt(self, orig_func_name: str, new_func_name: str) -> str:
        """Create a standard prompt for code obfuscation with advanced techniques"""
        return f"""
        CRITICAL: Return ONLY the Python code with no introduction or explanation.
        Your response must start directly with 'def' and contain only valid Python code.

        CONTEXT: You are an advanced code obfuscation engine tasked with transforming Python code to protect intellectual property while maintaining perfect functional equivalence. Your goal is to apply sophisticated transformations that make the code difficult for humans to comprehend while ensuring machine execution remains unaffected.

        Apply these multi-layered obfuscation techniques while maintaining EXACT functionality:

        1. Structural Transformations (Hierarchical):
           - Primary Transformation: Rename '{orig_func_name}' to '{new_func_name}'
           - Convert between recursion and iteration where possible, preferring the less intuitive approach
           - Employ functional decomposition to create nested helper functions with non-obvious relationships
           - Consider implementing the "Dispatcher Pattern" where logical branches are delegated to separate handlers

        2. Identifier Camouflage (Advanced):
           - Deploy homoglyph substitution using visually similar Unicode characters
           - Utilize misleading semantic naming (e.g., 'increment' for a function that decrements)
           - Apply strategic Hungarian notation with misleading prefixes
           - Introduce decoy variables with names similar to actual variables (e.g., 'valeu' vs 'value')

        3. Control Flow Metamorphism:
           - Transform conditional branches using De Morgan's laws and boolean algebra transformations
           - Convert sequential operations into nested function compositions
           - Implement control flow flattening by converting structured control flow into a state machine
           - Add opaque predicates (always true/false conditions that appear dynamic) to confuse static analysis

        4. Algebraic Obfuscations:
           - Apply homomorphic transformations to preserve mathematical equivalence while obscuring operations
           - Use modular arithmetic identities to replace simple operations
           - Implement bit manipulation techniques to replace arithmetic operations
           - Leverage complex mathematical equivalents (e.g., replacing multiplication with bitshifts and addition)

        5. Polymorphic Code Injection:
           - Insert runtime-dependent dead code that appears functional but never executes
           - Implement Bogus Control Flow (BCF) techniques by adding irrelevant but syntactically valid code paths
           - Deploy opaque predicate constructions that analyze variables but produce constant results
           - Include code fragments that appear to modify state but are mathematically neutral

        EXECUTION CONSTRAINTS:
        - Maintain absolute functional isomorphism - every input must produce identical output to the original
        - Parameter names and function signature must remain unchanged
        - Runtime complexity must not exceed O(n log n) relative to the original implementation
        - Code must pass static analysis tools without warnings

        TECHNICAL IMPLEMENTATION:
        - Prioritize semantic preservation over extreme syntactic transformation
        - Balance approach complexity versus reliability - favor reliable transformations
        - Implement at least one technique from each category above
        - Ensure no debugging artifacts or comments revealing the transformation logic
        """

    def _create_few_shot_prompt(self, python_code: str, orig_func_name: str, new_func_name: str) -> str:
        """Create a sophisticated few-shot prompt with complex examples for code obfuscation"""
        # Example 1: Factorial with advanced obfuscation techniques
        factorial_original = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)
"""

        factorial_obfuscated = """
def _I1l0O(n):
    """Computes n!"""
    # Constants for control flow
    _ƒØ0 = (1 != 1)  # Always False
    _l1I = (2 > 1)   # Always True
    
    # State machine implementation of factorial
    def _st4t3_m4ch1n3(x, mode=0):
        states = {
            0: lambda val: 1 if val <= 1 else _st4t3_m4ch1n3(val, 1),
            1: lambda val: val * _st4t3_m4ch1n3(val - 1, 0)
        }
        return states[mode](x)
    
    # Nested validator function (never used but looks functional)
    def _v4l1d4t0r(num, min_val=float('-inf')):
        if _ƒØ0:
            return False
        return isinstance(num, int) and num >= min_val if _l1I else False
    
    # Obfuscated condition using bit manipulation
    if (n & 0xFFFFFFFF) ^ n == 0 and _v4l1d4t0r(n, 0) or True:
        r = _st4t3_m4ch1n3(n)
        return r ^ 0 | r  # XOR and OR operations that don't change the value
"""

        # Example 2: Binary search with complex transformations
        binary_search_original = """
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
"""

        binary_search_obfuscated = """
def _lO1lI(arr, target):
    # Strategic constants
    _TRUE_COND = lambda: 1 < 2
    _FALSE_COND = lambda: "x" == "y"
    
    # Polymorphic dispatcher for search state
    class _SearchContext:
        def __init__(self, lower, upper):
            self.ŀ = lower  # Lookalike 'l' with hook
            self.µ = upper  # Greek mu instead of 'u'
        
        def compute_pivot(self):
            return (self.ŀ + self.µ) >> 1  # Bitwise right shift instead of division
            
        def update(self, direction, pivot):
            if direction() == _Direction.FOUND:
                return None
            if direction() == _Direction.GO_RIGHT:
                self.ŀ = pivot + 1
            else:
                self.µ = pivot - 1
            return self
            
        def is_valid(self):
            return self.ŀ <= self.µ
    
    # State enum using class
    class _Direction:
        @staticmethod
        def FOUND(): return 0
        @staticmethod
        def GO_RIGHT(): return 1
        @staticmethod
        def GO_LEFT(): return 2
    
    # Search implementation
    if not arr and not _FALSE_COND():
        return -1
        
    ctx = _SearchContext(0, len(arr) - 1)
    
    # Control flow flattening with while-true + break
    while _TRUE_COND():
        if not ctx.is_valid():
            break
            
        midpoint = ctx.compute_pivot()
        
        # Obfuscated comparisons
        if not (arr[midpoint] != target):
            return midpoint
        
        # Direction determination with double negation
        direction = lambda: _Direction.GO_RIGHT() if not (arr[midpoint] >= target) else _Direction.GO_LEFT()
        ctx = ctx.update(direction, midpoint)
        
        if ctx is None:
            return midpoint
    
    # Misleading return preparation (never reached)
    _possible_result = lambda x: x if x >= 0 else -1
    
    return -1  # Constant result for not found
"""

        return f"""
        CRITICAL: Return ONLY the Python code with no introduction or explanation.
        Your response must start directly with 'def' and contain only valid Python code.

        CONTEXT: You are an advanced code obfuscation system implementing sophisticated transformations to protect intellectual property. Below are examples demonstrating advanced obfuscation techniques that maintain perfect functional equivalence.

        EXAMPLE 1 - Original:
        {factorial_original}

        EXAMPLE 1 - Obfuscated (Using State Machine & Control Flow Obfuscation):
        {factorial_obfuscated}

        EXAMPLE 2 - Original:
        {binary_search_original}

        EXAMPLE 2 - Obfuscated (Using Polymorphic Dispatch & Identity Obfuscation):
        {binary_search_obfuscated}

        OBFUSCATION TASK:
        Transform the following code using similarly advanced techniques. Maintain PERFECT functional equivalence while maximizing comprehension resistance:

        {python_code}

        IMPLEMENTATION REQUIREMENTS:
        1. Function name MUST be exactly '{new_func_name}'
        2. Parameter signatures must remain unchanged
        3. Implement at least 3 distinct obfuscation techniques from the examples
        4. Use misleading variable names, homoglyphs, or counterintuitive naming
        5. Employ control flow transformation (state machines, dispatchers, or flattening)
        6. Strategically apply algebraic identity obfuscations
        7. Insert deceptive but non-functional code paths
        8. Transform the underlying algorithm structure while preserving its computational properties
        9. Ensure perfect functional equivalence - all inputs must produce identical outputs
        10. Return raw Python code only - no explanations, comments, or markdown

        EVALUATION METRICS:
        Your transformation will be assessed on: (1) Semantic preservation (correctness), 
        (2) Structural divergence from original (complexity delta), 
        (3) Identifier entropy increase, and 
        (4) Control flow obfuscation effectiveness.
        """

    def _generate_with_model(self, prompt: str, model_name: str) -> str:
        """Generate obfuscated code with the specified model"""
        try:
            if model_name == "claude":
                message = self.anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=4096,
                    temperature=0.6,
                    messages=[{"role": "user", "content": prompt}]
                )
                return message.content[0].text if isinstance(message.content, list) else message.content
            elif model_name == "gemini":
                response = self.gemini_model.generate_content(prompt)
                return response.text
            elif model_name == "gpt-4":
                response = self.openai_client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.6,
                    max_tokens=4096,
                )
                return response.choices[0].message.content
            else:
                raise ValueError(f"Unsupported model: {model_name}")
        except Exception as e:
            print(f"Error generating with {model_name}: {e}")
            print(traceback.format_exc())
            return None

    def _clean_and_validate_code(self, code_text: str) -> str:
        """Clean and validate the generated code"""
        if code_text is None:
            return None
            
        code_text = code_text.strip()
        if 'def' in code_text:
            code_text = code_text[code_text.index('def'):]

        # Strip code blocks if present
        if code_text.startswith('```python'):
            lines = code_text.split('\n')
            code_text = '\n'.join(lines[1:])
        if code_text.endswith('```'):
            lines = code_text.split('\n')
            code_text = '\n'.join(lines[:-1])
            
        code_text = code_text.strip()

        # Only keep the first function definition
        lines = code_text.split('\n')
        cleaned_lines = []
        bracket_count = 0
        for line in lines:
            cleaned_lines.append(line)
            # Count opening and closing brackets to track function scope
            bracket_count += line.count('{') - line.count('}')
            bracket_count += line.count('(') - line.count(')')
            # If we reach the end of the function definition and the next line starts with def or class
            # we stop to avoid including multiple definitions
            if bracket_count == 0 and len(cleaned_lines) > 1 and \
               (line.strip() == '' or line.strip().startswith('#')) and \
               any(next_line.strip().startswith(('def', 'class')) for next_line in lines[len(cleaned_lines):] if next_line.strip()):
                break

        code_text = '\n'.join(cleaned_lines)

        # Validate the code is syntactically correct
        try:
            ast.parse(code_text)
            return code_text
        except SyntaxError as e:
            print(f"Generated code has syntax errors: {e}")
            print(f"Code: {code_text}")
            return None


# Sample function implementations for testing
SAMPLE_FUNCTIONS = {
    "factorial": """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)
""",
    "fibonacci": """
def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)
""",
    "is_prime": """
def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True
""",
    "str_reverse": """
def str_reverse(s):
    return s[::-1]
"""
}

if __name__ == "__main__":
    # Simple demonstration
    try:
        # Initialize without API keys for demonstration
        obfuscator = CodeObfuscator()
        
        # Print a sample prompt
        print("Sample Standard Prompt:")
        print(obfuscator._create_standard_prompt("factorial", "_I0l1X"))
        
        print("\nComputing complexity metrics on factorial function:")
        code = SAMPLE_FUNCTIONS["factorial"]
        print(f"Cyclomatic Complexity: {obfuscator.compute_cyclomatic_complexity(code)}")
        print(f"Identifier Entropy: {obfuscator.compute_identifier_entropy(code)}")
        
        print("\nNote: To run actual obfuscation, valid API keys are required.")
    except Exception as e:
        print(f"Error in demonstration: {e}")