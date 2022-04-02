<a href="https://colab.research.google.com/github/wesleybeckner/deka/blob/main/notebooks/solutions/SOLN_P3_Stock_Cutting.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Stock Cutting Part 3:<br> The Column Generation Method

<br>

---

<br>

In this project notebook we'll be combining our dynamic program from the knapsack problem with a strategy called the _column generation method_

<br>

---

## 1.0: Import Functions and Libraries


```python
from collections import Counter
from itertools import combinations

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



```python
_widths = [170, 280, 320]
W = 4000
max_unique_layouts = 3

seed_patterns(_widths, W)
```




    [[Counter({170: 23}), 90],
     [Counter({280: 14}), 80],
     [Counter({320: 12}), 160],
     [Counter({170: 12, 280: 7}), 0],
     [Counter({170: 16, 320: 4}), 0],
     [Counter({280: 12, 320: 2}), 0],
     [Counter({170: 12, 280: 7}), 0]]



## 2.0 The Restricted Master Problem (RMP)

first we create our naieve solutions (restrict 1 layout per pattern)


```python
q = [80, 50, 100]
widths = w = [4, 6, 7]
W = 15

patterns = seed_patterns(widths, W, max_unique_layouts=1)
patterns
```




    [[Counter({4: 3}), 3], [Counter({6: 2}), 3], [Counter({7: 2}), 1]]



Then we perform the linear programming task. 


```python
from scipy.optimize import linprog
from math import ceil
import numpy as np

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
print(X)
print(f"total doffs: {sum(X)}")
```

    [27, 25, 50]
    total doffs: 102


These values of `X` are the minimum doffs we need to fulfill our order quantities `q` given a set of patterns, `patterns`.

The trick we next employee, is we determine how costly each width is to our solution. We do this by solving the dual variables of the linear program. 

The dual of a given linear program (LP) is another LP that is derived from the original (the primal) LP. Algorithmically this looks like the following:

1. Each variable in the primal LP becomes a constraint in the dual LP
2. Each constraint in the primal LP becomes a variable in the dual LP
3. The objective direction is inversed – maximum in the primal becomes minimum in the dual and vice versa

Notice below we switch the parameter fields for `c` and `b_ub` (the coefficients of the linear objective function and the linear constraint vector). And we take the negative transpose of our system of equations `A_ub`. 


```python
dual_problem = linprog(c=rhs_ineq,
        A_ub=-np.array(lhs_ineq).T,
        b_ub=obj,
        method="revised simplex")
val = [i for i in dual_problem['x']]
val
```




    [0.3333333333333333, 0.5, 0.5]



Roughly, this outcome is similar to the number of doffs dedicated to each width, normalized by the quantity ordered for each width. (Note that this comparison is only approximately true but is meant to give a conceptual guide).


```python
[i/j for i, j in zip(X, q)]
```




    [0.3375, 0.5, 0.5]



## 3.0 The Column Generation Subproblem (CGSP)

Ok. So what was that dual variable stuff all about? We are going to use the dual variable to update the value of each width. That's right, the behavior of each width in reference to the final doff quantities, `X` is used to bias the knapsack problem to give us a pattern that gives preferential treatment to the troublesome widths!


```python
wt = [4, 6, 7]
W = 15
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
print(pattern)
value = Counter([val[i] for i in list(sack)])
print(value)

total = 0
for worth, multiple in value.items():
    total += worth * multiple
total > 1
```

    Counter({4: 2, 6: 1})
    Counter({0.3333333333333333: 2, 0.5: 1})





    True



The last conditional above, `total > ` is our criteria for adding the new width to the growing host of patterns to then send to the RMP. If the total worth of the knapsack is greater than 1, this means our RMP will return a new solution with the added pattern that will result in overall fewer doffs.


```python
patterns.append([pattern, None])
patterns
```




    [[Counter({4: 3}), 3],
     [Counter({6: 2}), 3],
     [Counter({7: 2}), 1],
     [Counter({4: 2, 6: 1}), None]]




```python
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
print(X)
```

    [0, 5, 50, 41]


we see that the total number of doffs is reduced from 102 to 96!


```python
print(f"total doffs: {sum(X)}")
```

    total doffs: 96



```python
dual_problem = linprog(c=rhs_ineq,
        A_ub=-np.array(lhs_ineq).T,
        b_ub=obj,
        method="revised simplex")
val = [i for i in dual_problem['x']]
val
```




    [0.25, 0.5, 0.5]




```python
wt = [4, 6, 7]
W = 15
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
print(pattern)
value = Counter([val[i] for i in list(sack)])
print(value)

total = 0
for worth, multiple in value.items():
    total += worth * multiple
total > 1
```

    Counter({4: 2, 6: 1})
    Counter({0.25: 2, 0.5: 1})





    False



In this case, the knapsack problem does not produce a knapsack with a value greater than 1, and so we discontinue our CGSP!


```python
total
```




    1.0



## 4.0 Functions

Let's bundle our code into some functions


```python
from scipy.optimize import linprog
from math import ceil
import numpy as np

def solveX(patterns, widths, q):
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
```

Starting over with the former example...

We seed our patterns with the naive solutions


```python
q = [80, 50, 100]
widths = w = [4, 6, 7]
W = 15

patterns = seed_patterns(widths, W, max_unique_layouts=1)
patterns
```




    [[Counter({4: 3}), 3], [Counter({6: 2}), 3], [Counter({7: 2}), 1]]



We solve the RMP


```python
X, val = solveX(patterns, widths, q)
print(sum(X))
print(X, val)
```

    102
    [27, 25, 50] [0.3333333333333333, 0.5, 0.5]


We solve the CGSP


```python
pattern, total = pack_knap(w, val, W)
print(pattern, total)
print(total > 1)
```

    Counter({4: 2, 6: 1}) 1.1666666666666665
    True


Since the value is greater than 1 we add the pattern to our linprog and solve the RMP again


```python
patterns.append([pattern, 0])
X, val = solveX(patterns, widths, q)
print(sum(X))
print(X, val)

pattern, total = pack_knap(w, val, W)
print(pattern, total)
print(total > 1)
```

    96
    [0, 5, 50, 41] [0.25, 0.5, 0.5]
    Counter({4: 2, 6: 1}) 1.0
    False


We exit when we can no longer find a pattern that would improve the RMP

## 5.0 All Together Now


```python
q = [80, 50, 100]
widths = w = [4, 6, 7]
W = 15

patterns = seed_patterns(widths, W, max_unique_layouts=1)

while True:
    X, val = solveX(patterns, widths, q)
    pattern, total = pack_knap(w, val, W)
    if total > 1:
        patterns.append([pattern, 0])
        continue
    break
    
print()
print(f"total doffs: {sum(X)}", end="\n\n")
for quant, pattern in zip(X, patterns):
    if quant > 0:
        print(f"{quant}, {pattern[0]}")
```

    
    total doffs: 96
    
    5, Counter({6: 2})
    50, Counter({7: 2})
    41, Counter({4: 2, 6: 1})

