"""Collection of the core mathematical operators used throughout the code base."""

import math
import numpy as np

from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:


# - mul
def mul(a: float, b: float) -> float:
    return a * b

# - id
def id(a: float) -> float:
    return a

# - add
def add(a: float, b: float) -> float:
    return a + b

# - neg
def neg(a: float) -> float:
    return float(-a)

# - lt
def lt(a: float, b: float) -> float:
    return float(a < b)

# - eq
def eq(a: float, b: float) -> float:
    return float(a == b)

# - max
def max(a: float, b: float) -> float:
    return a if a > b else b

# - is_close
def is_close(a: float, b: float) -> float:
    return float(abs(a - b) < 1e-2)

# - sigmoid
def sigmoid(a: float) -> float:
    if a >= 0:
        return 1 / (1 + math.exp(-a))
    else:
        return math.exp(a) / (1 + math.exp(a))

# - relu
def relu(a: float) -> float:
    return (a > 0) * a

# - log
def log(a: float) -> float:
    return np.log(a)

# - exp
def exp(a: float) -> float:
    return np.exp(a)

# - log_back
def log_back(a: float, b: float) -> float:
    return b / a

# - inv
def inv(a: float) -> float:
    return 1 / a

# - inv_back
def inv_back(a: float, b: float) -> float:
    return neg(b) / a ** 2

# - relu_back
def relu_back(a: float, b: float) -> float:
    return (a > 0) * b


# Small practice library of elementary higher-order functions.

# Implement the following core functions

# - map
def map(f: Callable, it: Iterable):
    return [f(x) for x in it]

# - zipWith
def zipWith(f: Callable, it1: Iterable, it2: Iterable):
    return [f(a, b) for a, b in zip(it1, it2)]

# - reduce
def reduce(func: Callable, arr: Iterable) -> float:
    x = arr[0] if len(arr) > 0 else 0
    for y in arr[1:]:
        x = func(x, y)
    return x


# Use these to implement

# - negList : negate a list
def negList(a):
    return map(neg, a)

# - addLists : add two lists together
def addLists(a, b):
    return zipWith(add, a, b)

# - sum: sum lists
def sum(a):
    return reduce(add, a)

# - prod: take the product of lists
def prod(a):
    return reduce(mul, a)
