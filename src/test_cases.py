"""
Test Case Generator for Code Obfuscation Evaluation
==================================================
This module provides comprehensive test suites for validating
the functional correctness of original and obfuscated code.

Author: Lorenzo De Tomasi
Institution: Università degli Studi dell'Aquila
"""

from typing import Dict, List, Any, Union, Tuple
import random

def generate_test_cases(function_name: str) -> List[Any]:
    """
    Generate comprehensive test suites with edge cases for robust function validation.
    
    This function returns appropriate test cases for each supported function type,
    including edge cases, boundary conditions, typical cases, and special inputs
    designed to thoroughly test functional correctness.
    
    Args:
        function_name: Name of the function to generate test cases for
        
    Returns:
        List of test inputs appropriate for the specified function
    """
    test_cases = {
        # ===== Mathematical Functions =====
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
        
        "lcm": [
            (3, 4),             # Basic case
            (6, 8),             # Common case
            (7, 13),            # Coprime numbers
            (12, 18),           # Common factors
            (5, 10),            # One divides the other
            (0, 5),             # Edge case: zero input
            (5, 0),             # Edge case: zero input
            (0, 0),             # Edge case: both zero
            (17, 23),           # Larger coprime
            (-6, 8),            # Negative value
            (6, -8),            # Negative value
            (-6, -8),           # Both negative
            (2**10, 2**15)      # Powers of 2
        ],
        
        "power": [
            (2, 3),             # Basic case
            (5, 0),             # Zero exponent
            (0, 5),             # Zero base
            (0, 0),             # Zero base and exponent
            (10, 2),            # Square
            (2, 10),            # Higher power
            (3, 3),             # Cube
            (-2, 3),            # Negative base, odd exponent
            (-2, 2),            # Negative base, even exponent
            (2, -3),            # Negative exponent
            (0.5, 2),           # Fractional base
            (2, 0.5),           # Fractional exponent
            (10, 100)           # Performance test (if applicable)
        ],
        
        "sqrt_newton": [
            0,                  # Zero
            1,                  # Unit value
            4,                  # Perfect square
            9,                  # Perfect square
            16,                 # Perfect square
            25,                 # Perfect square
            100,                # Larger perfect square
            2,                  # Irrational result
            3,                  # Irrational result
            10,                 # Irrational result
            0.25,               # Fractional input
            0.01,               # Small fractional input
            10000,              # Large value
            -1,                 # Invalid: negative input
            "not a number"      # Invalid: non-numeric
        ],
        
        # ===== Sorting and Searching Algorithms =====
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
        
        "merge_sort": [
            [1, 2, 3, 4, 5],                # Already sorted
            [5, 4, 3, 2, 1],                # Reverse sorted
            [3, 1, 4, 1, 5, 9, 2, 6],       # Random order with duplicates
            [9, 8, 7, 6, 5, 4, 3, 2, 1, 0], # Perfectly reversed
            [],                             # Empty array
            [1],                            # Single element
            [1, 1, 1, 1, 1],                # All identical
            ["a", "c", "b", "d"],           # Strings
            [10**6, 10**5, 10**4, 10**3]    # Large numbers
        ],
        
        "quick_sort": [
            [1, 2, 3, 4, 5],                # Already sorted
            [5, 4, 3, 2, 1],                # Reverse sorted
            [3, 1, 4, 1, 5, 9, 2, 6],       # Random order with duplicates
            [9, 8, 7, 6, 5, 4, 3, 2, 1, 0], # Perfectly reversed
            [],                             # Empty array
            [1],                            # Single element
            [1, 1, 1, 1, 1],                # All identical
            ["a", "c", "b", "d"],           # Strings
            # Quick sort worst-case pivot scenario
            list(range(1000)) + [0]         # Nearly sorted with one outlier
        ],
        
        "insertion_sort": [
            [1, 2, 3, 4, 5],                # Already sorted
            [5, 4, 3, 2, 1],                # Reverse sorted
            [3, 1, 4, 1, 5, 9, 2, 6],       # Random order with duplicates
            [9, 8, 7, 6, 5, 4, 3, 2, 1, 0], # Perfectly reversed
            [],                             # Empty array
            [1],                            # Single element
            [1, 1, 1, 1, 1],                # All identical
            [float('inf'), 5, 4, 3, 2, 1],  # With infinity
            [-float('inf'), 1, 2, 3, 4, 5]  # With negative infinity
        ],
        
        "linear_search": [
            ([1, 2, 3, 4, 5], 1),           # Find first element
            ([1, 2, 3, 4, 5], 3),           # Find middle element
            ([1, 2, 3, 4, 5], 5),           # Find last element
            ([1, 2, 3, 4, 5], 0),           # Element not in array
            ([1, 2, 3, 4, 5], 6),           # Element not in array
            ([1, 1, 1, 1, 1], 1),           # Multiple matches (should return first)
            ([], 5),                        # Empty array
            ([5], 5),                       # Single element, found
            ([5], 7),                       # Single element, not found
            ([None, 1, 2, 3], None),        # Search for None
            ([1, 2, 3, 4, 5], "a"),         # Type mismatch
            (list(range(10000)), 9999)      # Large array, worst case
        ],
        
        # ===== String Manipulation =====
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
            "😊🙂🥳",                         # Unicode emojis
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
        ],
        
        "word_count": [
            "",                             # Empty string
            "hello",                        # Single word
            "hello world",                  # Two words
            "  spaces  around  ",           # Extra spaces
            "one,two,three",                # No spaces, with commas
            "line\nbreak",                  # With newline
            "tab\tseparated",               # With tab
            "multiple   spaces",            # Multiple spaces
            "punctuation! marks? here.",    # With punctuation
            "1 2 3 4 5",                    # Numbers as words
            "semi;colon:separated"          # Semicolons and colons
        ],
        
        "longest_common_substring": [
            ("abcde", "abfce"),             # Overlap at beginning
            ("", "abc"),                    # Empty string
            ("xyz", "abc"),                 # No common substring
            ("programming", "computer"),    # Partial overlap
            ("abcdefg", "bcd"),             # Substring
            ("abcabcabc", "abcabc"),        # Repeated patterns
            ("xyzabcxyz", "abcxyzabc"),     # Multiple common substrings
            ("a", "a"),                     # Single character
            ("a", "b"),                     # No match with single chars
            ("aaa", "aa"),                  # Repeated characters
            ("abc", "ABC"),                 # Case difference (case sensitive)
            ("The quick brown fox", "quick brown"), # Words in sentence
            ("abcdefghijklmnopqrstuvwxyz", "mnopqrstuv") # Long strings
        ],
        
        "levenshtein_distance": [
            ("kitten", "sitting"),          # Classic example
            ("hello", "hallo"),             # Single substitution
            ("", "abc"),                    # Empty to string
            ("abc", ""),                    # String to empty
            ("abc", "abc"),                 # Identical strings
            ("a", "b"),                     # Single character substitution
            ("Saturday", "Sunday"),         # Substring with changes
            ("thou shalt not", "you should not"), # Word replacements
            ("abcdef", "fedcba"),           # Reversed
            ("AAAA", "aaaa"),               # Case difference
            ("1234", "5678"),               # All different digits
            ("a very long string with many words", "short string few words"), # Very different lengths
            ("\n\t", " "),                  # Whitespace characters
            ("   ", " ")                    # Different amount of spaces
        ],
        
        "count_vowels": [
            "",                             # Empty string
            "hello",                        # Basic word (2 vowels)
            "aeiou",                        # All vowels
            "AEIOU",                        # All uppercase vowels
            "rhythm",                       # No vowels
            "Python programming",           # Mixed with spaces
            "AEIOUaeiou",                   # All vowels, mixed case
            "a e i o u",                    # Vowels with spaces
            "123!@#",                       # No vowels, special chars
            "ümlaut préférences",           # Accented vowels (may or may not count)
            "\t\n\r",                       # Just whitespace
            "aaaaa"                         # Repeated vowels
        ],
        
        # ===== Data Structure Functions =====
        "flatten_list": [
            [],                              # Empty list
            [1, 2, 3],                       # Already flat
            [1, [2, 3]],                     # Simple nested
            [1, [2, [3, 4]]],                # Multi-level nesting
            [[1, 2], [3, 4], [5, 6]],        # Multiple sublists
            [[[1, 2, 3]]],                   # Deeply nested
            [[], [[], []]],                  # Empty nested lists
            [1, 2, [], 3, [4, []]],          # Mixed with empty lists
            [[["a", "b"], [["c"]]], "d"],    # Mixed types
            [1, 2, 3, [4, 5, [6, 7, 8]]],    # Complex nesting
            # Very deep nesting - stress test
            [[[[[[[[[[[1]]]]]]]]]]],
            # Mixture of different types
            [1, "string", [2.5, [True, None]]]
        ],
        
        "list_permutations": [
            [],                              # Empty list
            [1],                             # Single element
            [1, 2],                          # Two elements
            [1, 2, 3],                       # Three elements
            ["a", "b"],                      # Strings
            [1, 1, 2],                       # Duplicates
            [1, 2, 3, 4],                    # Four elements (24 permutations)
            ["a", 1, True],                  # Mixed types
            [None],                          # None
            # Note: limiting to smaller lists as factorial growth makes large lists impractical
        ],
        
        "dict_merge": [
            ({"a": 1}, {"b": 2}),            # Non-overlapping keys
            ({"a": 1}, {"a": 2}),            # Overlapping keys
            ({}, {"a": 1}),                  # Empty first dict
            ({"a": 1}, {}),                  # Empty second dict
            ({}, {}),                        # Both empty
            ({"a": {"b": 1}}, {"a": {"c": 2}}), # Nested with merge
            ({"a": 1, "b": 2}, {"c": 3, "d": 4}), # Multiple keys
            ({"a": [1, 2]}, {"a": [3, 4]}),  # Lists as values
            ({"a": 1}, {"a": {"nested": 2}}), # Different value types
            ({"a": None}, {"a": 1}),         # None value
            ({"a": {"b": {"c": 1}}}, {"a": {"b": {"d": 2}}}), # Deep nesting
            # Complex case with multiple types
            (
                {"a": 1, "b": [1, 2], "c": {"d": 3}},
                {"a": 2, "b": [3], "c": {"e": 4}, "f": 5}
            )
        ],
        
        "remove_duplicates": [
            [],                              # Empty list
            [1, 2, 3],                       # No duplicates
            [1, 1, 2, 2, 3, 3],              # Adjacent duplicates
            [1, 2, 3, 1, 2, 3],              # Non-adjacent duplicates
            ["a", "b", "a", "c", "b"],       # String duplicates
            [1, "1"],                        # Different types that look same
            [None, None, 1, None],           # None values
            [True, 1, False, 0],             # Boolean/int equivalence
            [1.0, 1],                        # Float/int equivalence
            [[1], [1], [2]],                 # Lists as elements (may not work with all impls)
            [1, 1, 1, 1, 1],                 # All duplicates
            list(range(1000)) + list(range(500)), # Large list with duplicates
        ],
        
        "rotate_array": [
            ([1, 2, 3, 4, 5], 1),            # Basic right rotation
            ([1, 2, 3, 4, 5], 2),            # Multiple positions
            ([1, 2, 3, 4, 5], 0),            # Zero rotation
            ([1, 2, 3, 4, 5], 5),            # Full rotation (back to original)
            ([1, 2, 3, 4, 5], 6),            # More than length
            ([1, 2, 3, 4, 5], -1),           # Negative rotation (left)
            ([1, 2, 3, 4, 5], -2),           # Multiple left
            ([1, 2, 3, 4, 5], 100),          # Very large rotation
            ([], 3),                         # Empty array
            ([1], 10),                       # Single element
            ([1, 1, 1], 1),                  # All identical
            (["a", "b", "c"], 1),            # Strings
            ([None, 1, 2], 1),               # With None
        ],
        
        # ===== Recursive Algorithms =====
        "tower_of_hanoi": [
            1,                               # One disk
            2,                               # Two disks
            3,                               # Three disks
            4,                               # Four disks
            5,                               # Five disks
            0,                               # Edge case: zero disks
            -1,                              # Invalid: negative
            8,                               # Larger case (2^8 - 1 = 255 moves)
            # Note: larger values make tests very slow due to 2^n-1 moves
        ],
        
        "binary_tree_depth": [
            {"value": 1, "left": None, "right": None},  # Leaf node only
            {"value": 1, "left": {"value": 2, "left": None, "right": None}, "right": None},  # Left child only
            {"value": 1, "left": {"value": 2, "left": None, "right": None}, "right": {"value": 3, "left": None, "right": None}},  # Both children
            {"value": 1, "left": {"value": 2, "left": {"value": 4, "left": None, "right": None}, "right": None}, "right": {"value": 3, "left": None, "right": None}},  # Deeper left
            {"value": 1, "left": {"value": 2, "left": None, "right": None}, "right": {"value": 3, "left": None, "right": {"value": 5, "left": None, "right": None}}},  # Deeper right
            None,  # Empty tree
            # Balanced tree of depth 3
            {
                "value": 1,
                "left": {
                    "value": 2,
                    "left": {"value": 4, "left": None, "right": None},
                    "right": {"value": 5, "left": None, "right": None}
                },
                "right": {
                    "value": 3,
                    "left": {"value": 6, "left": None, "right": None},
                    "right": {"value": 7, "left": None, "right": None}
                }
            },
            # Skewed tree
            {
                "value": 1,
                "left": None,
                "right": {
                    "value": 2,
                    "left": None,
                    "right": {
                        "value": 3,
                        "left": None,
                        "right": {
                            "value": 4,
                            "left": None,
                            "right": None
                        }
                    }
                }
            }
        ],
        
        "flood_fill": [
            ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], 0, 0, 1),  # Fill entire grid
            ([[1, 1, 1], [1, 1, 0], [1, 0, 1]], 1, 1, 2),  # Fill subset
            ([[1, 1, 1], [1, 1, 0], [1, 0, 1]], 2, 2, 3),  # Fill isolated cell
            ([[1, 1, 1], [1, 1, 0], [1, 0, 1]], 0, 0, 2),  # Fill from corner
            ([[2, 2, 2], [2, 2, 2], [2, 2, 2]], 0, 0, 2),  # No change (same color)
            ([[]], 0, 0, 1),  # Empty row
            ([], 0, 0, 1),  # Empty grid
            ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 1, 1, 0),  # All different values
            ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], -1, 0, 1),  # Invalid position
            ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], 0, 10, 1),  # Out of bounds
            # Larger grid
            ([
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 1, 1, 1, 1, 0, 0, 1],
                [1, 0, 0, 1, 1, 1, 1, 0, 0, 1],
                [1, 0, 0, 1, 1, 1, 1, 0, 0, 1],
                [1, 0, 0, 1, 1, 1, 1, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            ], 4, 4, 2)  # Fill center region
        ],
        
        "knapsack": [
            ([], [], 10),                     # Empty lists
            ([1, 2, 3], [6, 10, 12], 5),      # Basic case
            ([1, 3, 4, 5], [1, 4, 5, 7], 7),  # Multiple possible combinations
            ([2, 3, 5, 7], [2, 5, 8, 10], 10), # Exact fit possible
            ([5, 10, 15, 20], [50, 60, 90, 100], 30), # Value maximization
            ([5], [10], 4),                   # Cannot fit
            ([10], [500], 10),                # Exact capacity
            ([1, 2, 3], [10, 10, 10], 6),     # Equal values
            ([5, 5, 5], [10, 20, 30], 10),    # Equal weights
            ([0], [0], 10),                   # Zero weight and value
            ([0, 2, 3], [5, 10, 15], 5),      # Item with zero weight
            ([1, 2, 3], [0, 10, 15], 5),      # Item with zero value
            (list(range(1, 20)), list(range(20, 39)), 30)  # Larger dataset
        ],
        
        "edit_distance": [
            ("kitten", "sitting"),            # Classic example
            ("hello", "hallo"),               # Single substitution
            ("", "abc"),                      # Empty to string
            ("abc", ""),                      # String to empty
            ("abc", "abc"),                   # Identical strings
            ("a", "b"),                       # Single character substitution
            ("Saturday", "Sunday"),           # Substring with changes
            ("thou shalt not", "you should not"), # Word replacements
            ("abcdef", "fedcba"),             # Reversed
            ("AAAA", "aaaa"),                 # Case difference
            ("1234", "5678"),                 # All different digits
            ("a very long string with many words", "short string few words"), # Very different lengths
            ("\n\t", " "),                    # Whitespace characters
            ("   ", " "),                     # Different amount of spaces
            # Very long strings
            ("abcdefghijklmnopqrstuvwxyz", "zyxwvutsrqponmlkjihgfedcba")
        ],
        
        "coin_change": [
            ([1, 5, 10], 12),                 # Classic example
            ([1, 2, 5], 11),                  # Another common case
            ([2, 5, 10], 3),                  # Not possible
            ([1, 3, 4], 6),                   # Multiple solutions
            ([5, 10, 25], 30),                # Exact multiples
            ([1], 100),                       # Single coin
            ([2], 3),                         # Impossible with restricted coins
            ([], 5),                          # No coins
            ([1, 2, 5], 0),                   # Zero amount
            ([2, 3, 5], 1),                   # Amount less than smallest coin
            ([3, 5, 7], 4),                   # No exact solution
            ([186, 419, 83, 408], 6249),      # Large values
            # Interesting case: large amount
            ([1, 5, 10, 25, 50], 9999)
        ]
    }
    
    return test_cases.get(function_name, [])


def generate_random_test_cases(function_name: str, count: int = 10) -> List[Any]:
    """
    Generate random test cases for a given function.
    Useful for stress testing or when specific test cases don't exist.
    
    Args:
        function_name: Name of the function to generate random tests for
        count: Number of random test cases to generate
        
    Returns:
        List of random test inputs appropriate for the function
    """
    random_generators = {
        # Mathematical functions
        "factorial": lambda: random.randint(0, 15),
        
        "fibonacci": lambda: random.randint(0, 25),
        
        "is_prime": lambda: random.randint(0, 1000),
        
        "gcd": lambda: (random.randint(1, 1000), random.randint(1, 1000)),
        
        "lcm": lambda: (random.randint(1, 100), random.randint(1, 100)),
        
        # Sorting and searching
        "bubble_sort": lambda: [random.randint(0, 100) for _ in range(random.randint(0, 20))],
        
        "binary_search": lambda: (
            sorted([random.randint(0, 100) for _ in range(random.randint(0, 20))]),
            random.randint(0, 100)
        ),
        
        # String manipulation
        "str_reverse": lambda: ''.join(
            random.choice('abcdefghijklmnopqrstuvwxyz ') 
            for _ in range(random.randint(0, 30))
        ),
        
        # Default generator for unspecified functions
        "default": lambda: random.randint(0, 100)
    }
    
    generator = random_generators.get(function_name, random_generators["default"])
    return [generator() for _ in range(count)]


if __name__ == "__main__":
    # Demonstration
    print("Test case examples:")
    for func in ["factorial", "binary_search", "str_reverse"]:
        cases = generate_test_cases(func)
        print(f"\n{func} ({len(cases)} cases):")
        print(cases[:3], "..." if len(cases) > 3 else "")
        
    print("\nRandom test cases:")
    print(generate_random_test_cases("factorial", 5))