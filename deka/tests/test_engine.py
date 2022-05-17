from collections import Counter
import numpy as np
from deka.engine import *


def test_small_bag():
    # the problem parameters
    val = [60, 50, 70, 30]
    wt = [5, 3, 4, 2]
    W = 5

    # the known solution
    max_val = 80
    max_items = [50, 30]

    t = initt(W, val)
    best = knapsack(wt, val, W, len(val), t)
    sack = reconstruct(len(val), W, t, wt)
    pattern = Counter([val[i] for i in list(sack)])

    assert best == max_val, "Optimal value not found"
    print("Optimal value found")

    assert list(pattern.keys()) == max_items, "Optimal items not found"
    print("Optimal items found")
    
def test_val_weight_equality():
    # the problem parameters
    val = wt = [2, 2, 2, 2, 5, 5, 5, 5]
    W = 14

    # the known solution
    max_val = 14
    max_items = Counter([5, 5, 2, 2])

    t = initt(W, val)
    best = knapsack(wt, val, W, len(val), t)
    sack = reconstruct(len(val), W, t, wt)
    pattern = Counter([val[i] for i in list(sack)])

    assert best == max_val, "Optimal value not found"
    print("Optimal value found")

    assert pattern == max_items, "Optimal items not found"
    print("Optimal items found")

def test_simple_stock_cutting():
    q = [80, 50, 100]
    widths = w = [4, 6, 7]
    W = 15
    ans = 96

    patterns = seed_patterns(widths, W, max_unique_layouts=1)

    while True:
        X, val = solveX(patterns, widths, q)
        pattern, total = pack_knap(w, val, W)
        if total > 1:
            patterns.append([pattern, 0])
            continue
        break
         
    assert sum(X) == ans, "Optimal doffs not found"
    print("test passed")
    print(f"total doffs: {sum(X)}", end="\n\n")
    for quant, pattern in zip(X, patterns):
        if quant > 0:
            print(f"{quant}, {pattern[0]}")
            
def test_stock_cutting_2():
    widths = [i+j for i, j in zip ([2, 2, 2], [170, 234, 158])]
    q = [879, 244, 181]
    W = 4160
    ans = 59

    patterns = seed_patterns(widths, W, max_unique_layouts=1)

    while True:
        X, val = solveX(patterns, widths, q)
        pattern, total = pack_knap(widths, val, W)
        if total > 1:
            patterns.append([pattern, 0])
            continue
        break

    assert sum(X) == ans, "Optimal doffs not found"
    print("test passed")
    print(f"total doffs: {sum(X)}", end="\n\n")
    for quant, pattern in zip(X, patterns):
        if quant > 0:
            print(f"{quant}, {pattern[0]}")
