"""
Sample functions for code obfuscation experiments.

This module contains various algorithmic patterns that represent
common computational tasks found in real-world software, grouped by category:

1. Mathematical Functions:
   - factorial: recursive function
   - fibonacci: recursive with overlapping subproblems
   - is_prime: conditional logic
   - gcd: Euclidean algorithm
   - lcm: least common multiple
   - power: recursive exponentiation
   - sqrt_newton: Newton's method for square root

2. Sorting and Searching Algorithms:
   - bubble_sort: nested iterations
   - binary_search: divide-and-conquer algorithm
   - merge_sort: divide-and-conquer with recursion
   - quick_sort: partition-based sorting
   - insertion_sort: iterative insertion
   - linear_search: simple iteration

3. String Manipulation:
   - str_reverse: simple string manipulation
   - is_palindrome: string testing
   - word_count: basic text processing
   - longest_common_substring: dynamic programming
   - levenshtein_distance: edit distance algorithm
   - count_vowels: character filtering

4. Data Structure Manipulation:
   - flatten_list: recursive list transformation
   - list_permutations: combinatorial algorithm
   - dict_merge: nested structure merging
   - remove_duplicates: set operations
   - rotate_array: array manipulation

5. Recursive Algorithms:
   - tower_of_hanoi: classic recursion problem
   - binary_tree_depth: tree traversal
   - flood_fill: graph traversal
   - knapsack: optimization problem
   - edit_distance: string comparison
   - coin_change: dynamic programming
"""

# ----- MATHEMATICAL FUNCTIONS -----

def factorial(n):
    """
    Calculate the factorial of n using recursion.
    
    Args:
        n: A non-negative integer
        
    Returns:
        The factorial of n (n!)
        
    Examples:
        >>> factorial(5)
        120
        >>> factorial(0)
        1
    """
    if n <= 1:
        return 1
    return n * factorial(n - 1)


def fibonacci(n):
    """
    Calculate the nth Fibonacci number using recursion.
    
    Args:
        n: A non-negative integer
        
    Returns:
        The nth Fibonacci number
        
    Examples:
        >>> fibonacci(0)
        0
        >>> fibonacci(1)
        1
        >>> fibonacci(5)
        5
    """
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)


def is_prime(n):
    """
    Determine if a number is prime.
    
    Args:
        n: An integer to check
        
    Returns:
        Boolean indicating if n is prime
        
    Examples:
        >>> is_prime(7)
        True
        >>> is_prime(4)
        False
    """
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


def gcd(a, b):
    """
    Find the greatest common divisor using Euclidean algorithm.
    
    Args:
        a, b: Two integers
        
    Returns:
        The greatest common divisor of a and b
        
    Examples:
        >>> gcd(48, 18)
        6
        >>> gcd(17, 23)
        1
    """
    if b == 0:
        return a
    return gcd(b, a % b)


def lcm(a, b):
    """
    Find the least common multiple of two numbers.
    
    Args:
        a, b: Two integers
        
    Returns:
        The least common multiple of a and b
        
    Examples:
        >>> lcm(3, 4)
        12
        >>> lcm(6, 8)
        24
    """
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // gcd(a, b)


def power(base, exp):
    """
    Calculate power using recursive exponentiation.
    
    Args:
        base: The base number
        exp: The exponent
        
    Returns:
        base^exp
        
    Examples:
        >>> power(2, 3)
        8
        >>> power(5, 0)
        1
    """
    if exp == 0:
        return 1
    if exp < 0:
        return 1 / power(base, -exp)
    if exp % 2 == 0:
        return power(base * base, exp // 2)
    else:
        return base * power(base, exp - 1)


def sqrt_newton(n, epsilon=1e-10):
    """
    Calculate square root using Newton's method.
    
    Args:
        n: Number to find square root of
        epsilon: Precision threshold
        
    Returns:
        Square root of n
        
    Examples:
        >>> round(sqrt_newton(16), 10)
        4.0
        >>> round(sqrt_newton(2), 10)
        1.4142135624
    """
    if n < 0:
        raise ValueError("Cannot compute square root of negative number")
    if n == 0:
        return 0

    x = n  # Initial guess
    while True:
        y = (x + n / x) / 2
        if abs(y - x) < epsilon:
            return y
        x = y


# ----- SORTING AND SEARCHING ALGORITHMS -----

def bubble_sort(arr):
    """
    Sort an array using bubble sort algorithm.
    
    Args:
        arr: List of comparable elements
        
    Returns:
        Sorted list
        
    Examples:
        >>> bubble_sort([3, 1, 4, 1, 5])
        [1, 1, 3, 4, 5]
    """
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr


def binary_search(arr, target):
    """
    Search for a target in a sorted array using binary search.
    
    Args:
        arr: Sorted list of comparable elements
        target: Element to find
        
    Returns:
        Index of target if found, -1 otherwise
        
    Examples:
        >>> binary_search([1, 2, 3, 4, 5], 3)
        2
        >>> binary_search([1, 2, 3, 4, 5], 6)
        -1
    """
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


def merge_sort(arr):
    """
    Sort an array using merge sort algorithm.
    
    Args:
        arr: List of comparable elements
        
    Returns:
        Sorted list
        
    Examples:
        >>> merge_sort([3, 1, 4, 1, 5])
        [1, 1, 3, 4, 5]
    """
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return merge(left, right)

def merge(left, right):
    """Helper function for merge_sort."""
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result


def quick_sort(arr):
    """
    Sort an array using quick sort algorithm.
    
    Args:
        arr: List of comparable elements
        
    Returns:
        Sorted list
        
    Examples:
        >>> quick_sort([3, 1, 4, 1, 5])
        [1, 1, 3, 4, 5]
    """
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quick_sort(left) + middle + quick_sort(right)


def insertion_sort(arr):
    """
    Sort an array using insertion sort algorithm.
    
    Args:
        arr: List of comparable elements
        
    Returns:
        Sorted list
        
    Examples:
        >>> insertion_sort([3, 1, 4, 1, 5])
        [1, 1, 3, 4, 5]
    """
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr


def linear_search(arr, target):
    """
    Search for a target in an array using linear search.
    
    Args:
        arr: List of comparable elements
        target: Element to find
        
    Returns:
        Index of target if found, -1 otherwise
        
    Examples:
        >>> linear_search([1, 2, 3, 4, 5], 3)
        2
        >>> linear_search([1, 2, 3, 4, 5], 6)
        -1
    """
    for i, item in enumerate(arr):
        if item == target:
            return i
    return -1


# ----- STRING MANIPULATION -----

def str_reverse(s):
    """
    Reverse a string.
    
    Args:
        s: String to reverse
        
    Returns:
        Reversed string
        
    Examples:
        >>> str_reverse("hello")
        'olleh'
    """
    return s[::-1]


def is_palindrome(s):
    """
    Check if a string is a palindrome.
    
    Args:
        s: String to check
        
    Returns:
        Boolean indicating if s is a palindrome
        
    Examples:
        >>> is_palindrome("racecar")
        True
        >>> is_palindrome("hello")
        False
    """
    s = ''.join(c.lower() for c in s if c.isalnum())
    return s == s[::-1]


def word_count(text):
    """
    Count the number of words in a text.
    
    Args:
        text: String to count words in
        
    Returns:
        Number of words
        
    Examples:
        >>> word_count("hello world")
        2
    """
    if not text:
        return 0
    words = text.strip().split()
    return len(words)


def longest_common_substring(s1, s2):
    """
    Find the longest common substring between two strings.
    
    Args:
        s1, s2: Two strings to compare
        
    Returns:
        The longest common substring
        
    Examples:
        >>> longest_common_substring("abcde", "abfce")
        'ab'
    """
    if not s1 or not s2:
        return ""

    m = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
    longest, end_pos = 0, 0

    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if s1[i-1] == s2[j-1]:
                m[i][j] = m[i-1][j-1] + 1
                if m[i][j] > longest:
                    longest = m[i][j]
                    end_pos = i

    return s1[end_pos - longest:end_pos]


def levenshtein_distance(s1, s2):
    """
    Calculate the Levenshtein distance between two strings.
    
    Args:
        s1, s2: Two strings to compare
        
    Returns:
        The edit distance between s1 and s2
        
    Examples:
        >>> levenshtein_distance("kitten", "sitting")
        3
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def count_vowels(s):
    """
    Count the number of vowels in a string.
    
    Args:
        s: String to check
        
    Returns:
        Number of vowels
        
    Examples:
        >>> count_vowels("hello")
        2
    """
    vowels = "aeiouAEIOU"
    count = 0
    for char in s:
        if char in vowels:
            count += 1
    return count


# ----- DATA STRUCTURE MANIPULATION -----

def flatten_list(nested_list):
    """
    Flatten a nested list structure.
    
    Args:
        nested_list: A potentially nested list
        
    Returns:
        Flattened list
        
    Examples:
        >>> flatten_list([1, [2, 3], [4, [5, 6]]])
        [1, 2, 3, 4, 5, 6]
    """
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result


def list_permutations(lst):
    """
    Generate all permutations of a list.
    
    Args:
        lst: List of elements
        
    Returns:
        List of all permutations
        
    Examples:
        >>> list_permutations([1, 2])
        [[1, 2], [2, 1]]
    """
    if len(lst) <= 1:
        return [lst]

    result = []
    for i in range(len(lst)):
        current = lst[i]
        remaining = lst[:i] + lst[i+1:]

        for p in list_permutations(remaining):
            result.append([current] + p)

    return result


def dict_merge(dict1, dict2):
    """
    Recursively merge two dictionaries.
    
    Args:
        dict1, dict2: Two dictionaries to merge
        
    Returns:
        Merged dictionary
        
    Examples:
        >>> dict_merge({"a": 1}, {"b": 2})
        {'a': 1, 'b': 2}
        >>> dict_merge({"a": {"b": 1}}, {"a": {"c": 2}})
        {'a': {'b': 1, 'c': 2}}
    """
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = dict_merge(result[key], value)
        else:
            result[key] = value
    return result


def remove_duplicates(lst):
    """
    Remove duplicates from a list while preserving order.
    
    Args:
        lst: List of elements
        
    Returns:
        List with duplicates removed
        
    Examples:
        >>> remove_duplicates([1, 2, 3, 1, 2])
        [1, 2, 3]
    """
    return list(dict.fromkeys(lst))


def rotate_array(arr, k):
    """
    Rotate an array k positions to the right.
    
    Args:
        arr: Array to rotate
        k: Number of positions to rotate
        
    Returns:
        Rotated array
        
    Examples:
        >>> rotate_array([1, 2, 3, 4, 5], 2)
        [4, 5, 1, 2, 3]
    """
    if not arr:
        return arr

    k = k % len(arr)
    if k == 0:
        return arr

    return arr[-k:] + arr[:-k]


# ----- RECURSIVE ALGORITHMS -----

def tower_of_hanoi(n):
    """
    Solve the Tower of Hanoi puzzle for n disks.
    
    Args:
        n: Number of disks
        
    Returns:
        List of moves (source, target)
        
    Examples:
        >>> tower_of_hanoi(2)
        [('A', 'B'), ('A', 'C'), ('B', 'C')]
    """
    def move(n, source, target, auxiliary, moves):
        if n > 0:
            # Move n-1 disks from source to auxiliary
            move(n-1, source, auxiliary, target, moves)
            # Move disk n from source to target
            moves.append((source, target))
            # Move n-1 disks from auxiliary to target
            move(n-1, auxiliary, target, source, moves)

    moves = []
    move(n, 'A', 'C', 'B', moves)
    return moves


def binary_tree_depth(root):
    """
    Calculate the maximum depth of a binary tree.
    
    Args:
        root: Root node of a binary tree (as a dictionary)
        
    Returns:
        Maximum depth of the tree
        
    Examples:
        >>> binary_tree_depth({"value": 1, "left": {"value": 2, "left": None, "right": None}, "right": {"value": 3, "left": None, "right": None}})
        2
    """
    if root is None:
        return 0

    left_depth = binary_tree_depth(root.get('left'))
    right_depth = binary_tree_depth(root.get('right'))

    return max(left_depth, right_depth) + 1


def flood_fill(image, sr, sc, new_color):
    """
    Perform flood fill on an image starting from position (sr, sc).
    
    Args:
        image: 2D grid representing the image
        sr, sc: Starting row and column
        new_color: New color to fill
        
    Returns:
        Updated image
        
    Examples:
        >>> flood_fill([[1, 1, 1], [1, 1, 0], [1, 0, 1]], 1, 1, 2)
        [[2, 2, 2], [2, 2, 0], [2, 0, 1]]
    """
    rows, cols = len(image), len(image[0])
    original_color = image[sr][sc]

    if original_color == new_color:
        return image

    def dfs(r, c):
        if (r < 0 or r >= rows or c < 0 or c >= cols or
            image[r][c] != original_color):
            return

        image[r][c] = new_color

        dfs(r+1, c)
        dfs(r-1, c)
        dfs(r, c+1)
        dfs(r, c-1)

    dfs(sr, sc)
    return image


def knapsack(weights, values, capacity):
    """
    Solve the 0/1 knapsack problem.
    
    Args:
        weights: List of item weights
        values: List of item values
        capacity: Knapsack capacity
        
    Returns:
        Maximum value that can be obtained
        
    Examples:
        >>> knapsack([1, 3, 4, 5], [1, 4, 5, 7], 7)
        9
    """
    n = len(weights)

    # Base case: no items or no capacity
    if n == 0 or capacity == 0:
        return 0

    # If weight of the nth item is more than the capacity,
    # then it cannot be included
    if weights[n-1] > capacity:
        return knapsack(weights[:-1], values[:-1], capacity)

    # Return the maximum of two cases:
    # 1. nth item included
    # 2. nth item not included
    return max(
        values[n-1] + knapsack(weights[:-1], values[:-1], capacity - weights[n-1]),
        knapsack(weights[:-1], values[:-1], capacity)
    )


def edit_distance(s1, s2):
    """
    Calculate the edit distance between two strings.
    
    Args:
        s1, s2: Two strings to compare
        
    Returns:
        The edit distance between s1 and s2
        
    Examples:
        >>> edit_distance("kitten", "sitting")
        3
    """
    m, n = len(s1), len(s2)

    # Create a table to store results of subproblems
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    # Fill the table
    for i in range(m + 1):
        for j in range(n + 1):
            # If first string is empty, only option is to
            # insert all characters of second string
            if i == 0:
                dp[i][j] = j

            # If second string is empty, only option is to
            # remove all characters of first string
            elif j == 0:
                dp[i][j] = i

            # If last characters are the same, ignore them
            # and recur for remaining string
            elif s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]

            # If last characters are different, consider all
            # possibilities and find minimum
            else:
                dp[i][j] = 1 + min(
                    dp[i][j-1],      # Insert
                    dp[i-1][j],      # Remove
                    dp[i-1][j-1]     # Replace
                )

    return dp[m][n]


def coin_change(coins, amount):
    """
    Find the minimum number of coins to make up a given amount.
    
    Args:
        coins: List of coin denominations
        amount: Target amount
        
    Returns:
        Minimum number of coins needed
        
    Examples:
        >>> coin_change([1, 2, 5], 11)
        3
    """
    # Initialize the dp array with float('inf')
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0  # Base case

    # Fill the dp array
    for coin in coins:
        for x in range(coin, amount + 1):
            dp[x] = min(dp[x], dp[x - coin] + 1)

    return dp[amount] if dp[amount] != float('inf') else -1