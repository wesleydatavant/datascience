from collections import Counter
import numpy as np
from deka.engine import initt, knapsack, reconstruct


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
    pattern = Counter(np.array(val)[list(sack)])
    
    assert best == max_val, "Optimal value not found"
    print("Optimal value found")
    
    assert list(pattern.keys()) == max_items, "Optimal items not found"
    print("Optimal items found")

