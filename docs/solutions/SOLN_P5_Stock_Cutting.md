<a href="https://colab.research.google.com/github/wesleybeckner/deka/blob/main/notebooks/solutions/SOLN_P5_Stock_Cutting.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Stock Cutting Part 5:<br> Edge Cases and API

<br>

---

<br>

In this project notebook we'll be 

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
            if [pattern, loss] not in patterns:
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
```

run tests


```python
test_val_weight_equality()
print()
test_small_bag()
print()
test_simple_stock_cutting()
print()
test_stock_cutting_2()
```

    Optimal value found
    Optimal items found
    
    Optimal value found
    Optimal items found
    
    test passed
    total doffs: 96
    
    5, Counter({6: 2})
    50, Counter({7: 2})
    41, Counter({4: 2, 6: 1})
    
    test passed
    total doffs: 59
    
    39, Counter({172: 20, 160: 3, 236: 1})
    16, Counter({236: 12, 172: 4, 160: 4})
    4, Counter({172: 15, 236: 6, 160: 1})


## When we need to limit the number of patterns in the layouts

let's suppose we want a solution that only has 2 patterns in any given layout. When using the column generation method we used `seed_patterns` to create the naive layouts:


```python
q = [80, 50, 100]
widths = w = [4, 6, 7]
W = 15
ans = 96

patterns = seed_patterns(widths, W, max_unique_layouts=1)
patterns
```




    [[Counter({4: 3}), 3], [Counter({6: 2}), 3], [Counter({7: 2}), 1]]



Now, however, we'd like to include layouts that are combinations of each width. Let's start with just 2 layouts per pattern:


```python
q = [80, 50, 100]
widths = w = [4, 6, 7]
W = 15
ans = 96

patterns = seed_patterns(widths, W, max_unique_layouts=2)
patterns
```




    [[Counter({4: 3}), 3],
     [Counter({6: 2}), 3],
     [Counter({7: 2}), 1],
     [Counter({4: 2, 6: 1}), 1],
     [Counter({4: 2, 7: 1}), 0]]



The trailing numbers after the Counter object tell us the remainder on each pattern. We can now use these in our linear programming optimization step to determine how these layouts can be combined to fullfill the order.

Because we are not allowing the knapsack problem to deliver any amount of unique widths in a pattern, we will not be using the column generation method. Instead we send the layouts we've created to the linear programming optimization step and take the best answer we can get:


```python
X, val = solveX(patterns, widths, q)
assert sum(X) == ans, "Optimal doffs not found"
print("test passed")
print(f"total doffs: {sum(X)}", end="\n\n")
for quant, pattern in zip(X, patterns):
    if quant > 0:
        print(f"{quant}, {pattern[0]}")
```

    test passed
    total doffs: 96
    
    5, Counter({6: 2})
    50, Counter({7: 2})
    41, Counter({4: 2, 6: 1})


Incidentally the solution is the same because the column generation method solution did not have any patterns with more than 2 layouts. Let's take another example with the second unit test parameters:


```python
widths = [i+j for i, j in zip ([2, 2, 2], [170, 234, 158])]
q = [879, 244, 181]
W = 4160
ans = 59

patterns = seed_patterns(widths, W, max_unique_layouts=2)
display(patterns)
print()
X, val = solveX(patterns, widths, q)
assert sum(X) <= ans, "Optimal doffs not found"
print("test passed")
print(f"total doffs: {sum(X)}", end="\n\n")
for quant, pattern in zip(X, patterns):
    if quant > 0:
        print(f"{quant}, {pattern[0]}")
```


    [[Counter({172: 24}), 32],
     [Counter({236: 17}), 148],
     [Counter({160: 26}), 0],
     [Counter({172: 20, 236: 3}), 12]]


    
    test passed
    total doffs: 58
    
    7, Counter({236: 17})
    7, Counter({160: 26})
    44, Counter({172: 20, 236: 3})


In this case, we actually get an answer that is better than the column generation method!

## When we need to limit the number of layouts in a solution

When we need to limit the total number of layouts in a solution, we will have to make multiple calls to the linear programming step and compare the results of each call. 

Let's say we are not limited by the number of layouts per pattern. We would then want to send all permutations of the following to the linear programming step:


```python
widths = [i+j for i, j in zip ([2, 2, 2], [170, 234, 158])]
q = [879, 244, 181]
W = 4160
ans = 59

patterns = seed_patterns(widths, W, max_unique_layouts=3)
display(patterns)
```


    [[Counter({172: 24}), 32],
     [Counter({236: 17}), 148],
     [Counter({160: 26}), 0],
     [Counter({172: 20, 236: 3}), 12],
     [Counter({172: 4, 236: 12, 160: 4}), 0]]



```python
max_patterns = 2

# current_max will account for all max_patterns and anything less
for current_max in range(1, max_patterns+1):
    
    # our pre_sacks contains the collection of combinations
    pre_sacks = list(combinations(patterns, current_max))
    
    # the sack is what we will send to the lin prog program
    # we will borrow from column_gen to supplement any pattern 
    # combinations that could use an extra layout (to get to
    # the max_patterns value)
    for sack in pre_sacks:
        print(sack)
    print()
```

    ([Counter({172: 24}), 32],)
    ([Counter({236: 17}), 148],)
    ([Counter({160: 26}), 0],)
    ([Counter({172: 20, 236: 3}), 12],)
    ([Counter({236: 12, 172: 4, 160: 4}), 0],)
    
    ([Counter({172: 24}), 32], [Counter({236: 17}), 148])
    ([Counter({172: 24}), 32], [Counter({160: 26}), 0])
    ([Counter({172: 24}), 32], [Counter({172: 20, 236: 3}), 12])
    ([Counter({172: 24}), 32], [Counter({236: 12, 172: 4, 160: 4}), 0])
    ([Counter({236: 17}), 148], [Counter({160: 26}), 0])
    ([Counter({236: 17}), 148], [Counter({172: 20, 236: 3}), 12])
    ([Counter({236: 17}), 148], [Counter({236: 12, 172: 4, 160: 4}), 0])
    ([Counter({160: 26}), 0], [Counter({172: 20, 236: 3}), 12])
    ([Counter({160: 26}), 0], [Counter({236: 12, 172: 4, 160: 4}), 0])
    ([Counter({172: 20, 236: 3}), 12], [Counter({236: 12, 172: 4, 160: 4}), 0])
    


Taking those combinations into effect now:


```python
max_patterns = 2

# current_max will account for all max_patterns and anything less
for current_max in range(1, max_patterns+1):
    
    # our pre_sacks contains the collection of combinations
    pre_sacks = list(combinations(patterns, current_max))
    
    # the sack is what we will send to the lin prog program
    # we will borrow from column_gen to supplement any pattern 
    # combinations that could use an extra layout (to get to
    # the max_patterns value)
    for sack in pre_sacks:
        sack = list(sack)
        while True:
            X, val = solveX(sack, widths, q)
            pattern, total = pack_knap(widths, val, W)
            if (total > 1) and (len(sack) < max_patterns):
                sack.append([pattern, 0])
                continue
            break
        if sum(X) > 0:
            # print(total)
            # print(len(sack))
            print(f"total doffs: {sum(X)}")
            for quant, pattern in zip(X, sack):
                if quant > 0:
                    print(f"{quant}, {pattern[0]}")
            print()
```

    total doffs: 263
    82, Counter({172: 20, 236: 3})
    181, Counter({160: 1})
    
    total doffs: 76
    46, Counter({236: 12, 172: 4, 160: 4})
    30, Counter({172: 24})
    
    total doffs: 76
    30, Counter({172: 24})
    46, Counter({236: 12, 172: 4, 160: 4})
    
    total doffs: 220
    220, Counter({236: 12, 172: 4, 160: 4})
    
    total doffs: 89
    7, Counter({160: 26})
    82, Counter({172: 20, 236: 3})
    
    total doffs: 220
    220, Counter({236: 12, 172: 4, 160: 4})
    
    total doffs: 81
    35, Counter({172: 20, 236: 3})
    46, Counter({236: 12, 172: 4, 160: 4})
    


## When we need a single layout


```python
def make_best_pattern(q, w, n, usable_width=4160, verbiose=True):
    """
    Creates the best possible pattern such that all orders are fullfilled in a single
    layout

    Parameters
    ----------
    q: list
        rolls required (in jumbo lengths)
    w: list
        widths required
    n: list
        neckins for widths
    usable_width: int
        jumbo/doff usable width

    Returns
    -------
    layout: list
        cuts for jumbo for each width (no width is excluded)
    """

    # if not all slits can fit in a single bin, do not return a single optimum layout
    if np.sum([n,w]) > usable_width:
        return None

    layout = [max(1, math.floor(i/sum(q)*usable_width/j)) for i,j in zip(q,w)]


    # give priority to widths that had to round down the most
    # when filling up the rest of the pattern
    remainder = [math.remainder(i/sum(q)*usable_width/j, 1) if (math.remainder(i/sum(q)*usable_width/j, 1)
                                                        < 0) else -1 for i,j in zip(q,w) ]
    order = np.argsort(remainder)
    # sometimes the floor still puts us over
    while usable_width - sum([i*j for i,j in zip(layout,w)]) < 0:
        layout[np.argmax(layout)] -= 1

    while (usable_width - sum([i*j for i,j in zip(layout,w)])) > min(w):
        for i in order[::-1]:
            layout[i] += 1
            if usable_width - sum([i*j for i,j in zip(layout,w)]) < 0:
                layout[i] -= 1

    # compute the loss for the final layout
    layout_loss = usable_width - sum([i*j for i,j in zip(layout,w)])
    if verbiose:
        print("layout pattern: {}".format(dict(zip([i-j for i,j in zip(w,n)],layout))))
        print("pattern loss: {:0.2f} %".format(layout_loss/usable_width*100))

    # sometimes all orders can't be fullfilled in a single layout
    if any([i == 0 for i in layout]):
        return layout
    else:
        # multiply to get the minimum doffs required
        # layout * doffs > q
        doffs = max([math.ceil(i/j) for i,j in zip(q, layout)])
        if verbiose:
            print("minimum doffs to fill order: {}".format(doffs))

        # what inventory is created
        inventory = dict(zip([i-j for i,j in zip(w,n)],[i*doffs-j for i,j in zip(layout,q)]))
        if verbiose:
            print("inventory created: {}".format(inventory))

    return layout
```


```python
widths = [170, 234, 158]
n = [2, 2, 2]
q = [879, 244, 181]
W = 4160
ans = 59

import math
make_best_pattern(q, widths, n, usable_width=W, verbiose=True)
```

    layout pattern: {168: 16, 232: 3, 156: 4}
    pattern loss: 2.55 %
    minimum doffs to fill order: 82
    inventory created: {168: 433, 232: 2, 156: 147}





    [16, 3, 4]


