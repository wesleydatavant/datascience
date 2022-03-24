<a href="https://colab.research.google.com/github/wesleybeckner/deka/blob/main/notebooks/solutions/SOLN_P1_Stock_Cutting.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Stock Cutting Part 2:<br> Finding Good (But not Best) Patterns

<br>

---

<br>

In this project notebook we'll be leveraging our solution to the knapsack problem to create viable patterns for stock cutting.

<br>

---

## 1.0: Import Functions and Libraries


```python
import numpy as np
from collections import Counter

def initt(W, val):
    return [[None for i in range(W + 1)] for j in range(len(val) + 1)]

def knapsack(wt, val, w, n, t):
    # n, w will be the row, column of our table
    # solve the basecase. 
    if w == 0 or n == 0:
        return 0

    elif t[n][w] != None:
        return t[n][w]

    # now include the conditionals
    if wt[n-1] <= w:
        t[n][w] = max(
            knapsack(wt, val, w, n-1, t),
            knapsack(wt, val, w-wt[n-1], n-1, t) + val[n-1])
        return t[n][w]

    elif wt[n-1] > w:
        t[n][w] = knapsack(wt, val, w, n-1, t)
        return t[n][w]
    
def reconstruct(N, W, t, wt):
    recon = set()
    for j in range(N)[::-1]:
        if t[j+1][W] not in t[j]:
            recon.add(j)
            W = W - wt[j] # move columns in table lookup
        if W < 0:
            break
        else:
            continue
    return recon

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
    pattern = Counter(np.array(val)[list(sack)])

    assert best == max_val, "Optimal value not found"
    print("Optimal value found")

    assert pattern == max_items, "Optimal items not found"
    print("Optimal items found")
```


```python
test_small_bag()
```

    Optimal value found
    Optimal items found



```python
test_val_weight_equality()
```

    Optimal value found
    Optimal items found


## 1.1: Why good but not best?

The shortcoming of the knapsack problem is that while it is able to find the best possible configuration to maximize the value of a knapsack, it does not consider constraints around items we _must_ include. That is, when we create a schedule for our stock cutter, it is necessary that we deliver _all_ orders within a reasonable time. 

To over come this hurdle, we combine results from the knapsack problem (and any other pattern generative algorithm we would like to include) with a linear opimization task. We will cover the linear optimization task in a later notebook. Just know for now that we are still working on creating candidate patterns.
