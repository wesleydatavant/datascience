<a href="https://colab.research.google.com/github/wesleybeckner/deka/blob/main/notebooks/solutions/SOLN_P4_Stock_Cutting.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Stock Cutting Part 4:<br> Unit Tests

<br>

---

<br>

In this project notebook we'll be writing unit tests for our cutting stock algorithm

<br>

---

## 1.0: Import Functions and Libraries


```python
from collections import Counter
from itertools import combinations
from scipy.optimize import linprog
from math import ceil
import numpy as np

def solveX(patterns, widths, q):
    """
    solves the linprog (minimum doffs needed given set of patterns)
    as well as the dual problem
    """
    lhs_ineq = []
    for pattern in patterns:

        # inset will be our full build of a given "pattern"
        inset = []
        for width in widths:

            # try to access the slitwidth counts, otherwise
            # it means none of that slitwidth was included 
            try:
                inset.append(-pattern[0][width])
            except:
                inset.append(0)

        # add inset to the set of equations (patterns)        
        lhs_ineq.append(inset)
    lhs_ineq = np.array(lhs_ineq).T.tolist()

    # rhs is the min orders we need for each slitwidth
    rhs_ineq = [-i for i in q]

    # min x1 + x2 + .... Xn
    obj = np.ones(len(lhs_ineq[0]))

    # linprog will determine the minimum number we need
    # of each pattern
    result = linprog(c=obj,
            A_ub=lhs_ineq,
            b_ub=rhs_ineq,
            method="revised simplex")

    X = [ceil(i) for i in result['x']]
    
    
    dual_problem = linprog(c=rhs_ineq,
        A_ub=-np.array(lhs_ineq).T,
        b_ub=obj,
        method="revised simplex")
    val = [i for i in dual_problem['x']]
    
    return X, val

def pack_knap(wt, val, W):
    new_wt = []
    new_val = []
    for w, v in zip(wt, val):
        new_wt += [w]*int(W/w)
        new_val += [v]*int(W/w)
    wt = new_wt
    val = new_val
    t = initt(W, val)
    best = knapsack(wt, val, W, len(val), t)
    loss = W - best
    sack = reconstruct(len(val), W, t, wt)
    pattern = Counter([wt[i] for i in list(sack)])
    
    value = Counter([val[i] for i in list(sack)])
    

    total = 0
    for worth, multiple in value.items():
        total += worth * multiple
    return pattern, total

def seed_patterns(_widths, W, max_unique_layouts=3):
    patterns = []
    for current_max in range(1, max_unique_layouts+1):
        pre_sacks = list(combinations(_widths, current_max))
        for widths in pre_sacks:
            new = []
            for w in widths:
                new += [w]*int(W/w)
            widths = new

            t = initt(W, widths)
            best = knapsack(widths, widths, W, len(widths), t)
            loss = W - best
            sack = reconstruct(len(widths), W, t, widths)
            pattern = Counter([widths[i] for i in list(sack)])
            patterns.append([pattern, loss])
    return patterns

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
        if (t[j+1][W] not in t[j]) and (t[j+1][W] != 0):
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
```


```python
test_simple_stock_cutting()
```

    test passed
    total doffs: 96
    
    5, Counter({6: 2})
    50, Counter({7: 2})
    41, Counter({4: 2, 6: 1})



```python
(12*234 + 8*158)
```




    4072




```python
print(f"234: {17*7+3*44}")
print(f"170: {20*44}")
print(f"158: {26*7}")
```

    234: 251
    170: 880
    158: 182



```python
#  [879, 244, 181]
print(f"234: {(17*7+3*44)/244}")
print(f"170: {(20*44)/879}")
print(f"158: {(26*7)/181}")
```

    234: 1.028688524590164
    170: 1.0011376564277588
    158: 1.0055248618784531



```python
(((17*7+3*44)+ (20*44) + (26*7))
 /(244 + 879 + 181))*100
```




    100.69018404907975




```python
print(f"{4160-236*17}")
print(f"{4160-(20*172+3*236)}")
print(f"{4160-26*160}")
```

    148
    12
    0



```python
# q = [448, 931, 2179, 864]
# widths = [218, 170, 234, 208]
# W = 1500

widths = [i+j for i, j in zip ([2, 2, 2], [170, 234, 158])]
q = [879, 244, 181]
W = 4160

patterns = seed_patterns(widths, W, max_unique_layouts=1)

while True:
    X, val = solveX(patterns, widths, q)
    pattern, total = pack_knap(widths, val, W)
    if total > 1:
        patterns.append([pattern, 0])
        continue
    break

print("test passed")
print(f"total doffs: {sum(X)}", end="\n\n")
for quant, pattern in zip(X, patterns):
    if quant > 0:
        print(f"{quant}, {pattern[0]}")
```

    test passed
    total doffs: 59
    
    39, Counter({172: 20, 160: 3, 236: 1})
    16, Counter({236: 12, 172: 4, 160: 4})
    4, Counter({172: 15, 236: 6, 160: 1})



```python
(16*170 + 6*234)
```




    4124




```python
(14*170 + 11*158)
```




    4118




```python
# let's clean up these answers
# target is an input - needs ot be made an outpu (i.e. shows % above or below the order quantity)
# remove answers that are outside of +/- 10%

# show devops process on docker, cli, azure app service
```
