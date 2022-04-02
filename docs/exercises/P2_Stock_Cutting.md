<a href="https://colab.research.google.com/github/wesleybeckner/deka/blob/main/notebooks/exercises/P2_Stock_Cutting.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Stock Cutting Part 2:<br> Finding Good (But not Best) Patterns

<br>

---

<br>

In this project notebook we'll be leveraging our solution to the knapsack problem to create viable patterns for stock cutting.

<br>

---

## 1.0: Import Functions and Libraries


```python
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


## 1.1 Modifications from knapsack to cutting stock

You may have guessed this, but for all of our problems the `wt` list and `val` list will always be the same; they will be the list of widths scheduled to cut from stock. 

When we get our orders, we will need to adjust such that we are solving the _appropriate_ problem with the knapsack function. To give an example, we might have 100 orders to fulfill with slitwidth 170. However we can max only fit 20 on a roll. In this situation, we don't want to include all 100 repeat widths in the knapsack problem, because we know we can't possibly fit that many. Instead, we want to only provide the maximum number of 170's we could possibly fit on a roll. This will make the algorithm more efficient.

### 1.1.1 Can we simplify the knapsack function?


```python
wt = val = [170, 280, 320]
W = 4000
t = initt(W, val)
best = knapsack(wt, val, W, len(val), t)
sack = reconstruct(len(val), W, t, wt)
pattern = Counter([val[i] for i in list(sack)])
pattern
```




    Counter({170: 1, 280: 1, 320: 1})



### 🎒 Exercise 1: replace wt and val in `knapsack` and nonetype in `initt`

Notice how in the above cell we set `wt` and `val` equal to our product widths. In the cell below, rewrite the `knapsack` function so that `widths` takes the place of both `wt` and `val`


```python
def initt(W, val):
    return [[-1 for i in range(W + 1)] for j in range(len(val) + 1)]

def knapsack(widths, w, n, t):
    # n, w will be the row, column of our table
    # solve the basecase. 
    pass
```

Do the same thing for the `reconstruct` function. And while we're at it, let's change `N` to `n` and `W` to `w` so that our variables are consistent across both functions


```python
def reconstruct(n, w, t, widths):
    pass
```

and lets test our new functions


```python
widths = [170, 280, 320]
W = 4000
t = initt(W, widths)
best = knapsack(widths, W, len(widths), t)
sack = reconstruct(len(widths), W, t, widths)
pattern = Counter([widths[i] for i in list(sack)])
pattern
```




    Counter({170: 1, 280: 1, 320: 1})



### 1.1.2 How many slit widths?

Does our answer to the knapsack problem above make sense? It does based on what we fed the function. However, in reality what we're looking for is the best pattern given a list of unique slit widths even if that requires repeating units of slit widths. So how do we modify the way we call the function?


```python
widths = [170, 280, 320]
W = 4000
for w in widths:
    print([w]*int(W/w))
```

    [170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170]
    [280, 280, 280, 280, 280, 280, 280, 280, 280, 280, 280, 280, 280, 280]
    [320, 320, 320, 320, 320, 320, 320, 320, 320, 320, 320, 320]


### 🎒 Exercise 2: call knapsack with a modified list of widths

In the above, we created new lists that properly signify the maximum number of units we could fit into the stock width. It is this list of items that we wish to feed into our knapsack problem. Rewrite our call to the knapsack problem below

Also include neckin when you send this to knapsack


```python
widths = [170, 280, 320]
neckin = [5, 6, 7]
W = 4000
# modify call to knapsack

```

    3999





    Counter({175: 3, 327: 8, 286: 3})



### 🎒 Exercise 3: report the loss

As a last adjustment, we want to think of the loss from a pattern, not the total number of millimeters used. Calculate the loss


```python
widths = [170, 280, 320]
neckin = [5, 6, 7]
W = 4000
# modify call to knapsack

```

    1
    0.025





    Counter({175: 3, 327: 8, 286: 3})



## 1.2: Why good but not best?

The shortcoming of the knapsack problem is that while it is able to find the best possible configuration to maximize the value of a knapsack, it does not consider constraints around items we _must_ include. That is, when we create a schedule for our stock cutter, it is necessary that we deliver _all_ orders within a reasonable time. 

To over come this hurdle, we combine results from the knapsack problem (and any other pattern generative algorithm we would like to include) with a linear opimization task. We will cover the linear optimization task in a later notebook. Just know for now that we are still working on creating candidate patterns.

### 1.2.1 Find all unique combinations of slit widths


```python
widths = [170, 280, 320]
neckin = [5, 6, 7]
W = 4000
# modify call to knapsack
# this new list will represent what we actually send to the function knapsack
new = []
for w in widths:
    new += [w]*int(W/w)
widths = new
t = initt(W, widths)
best = knapsack(widths, W, len(widths), t)
loss = W - best
print(loss)
percent = loss/W*100
print(percent)
sack = reconstruct(len(widths), W, t, widths)
pattern = Counter([widths[i] for i in list(sack)])
pattern
```

    1
    0.025





    Counter({175: 3, 327: 8, 286: 3})



### 🎒 Exercise 4: permutate the list of unique widths


```python
from itertools import combinations
```


```python
_widths = [170, 280, 320]
W = 4000
max_unique_layouts = 3

def seed_patterns(_widths, W, max_unique_layouts=3):
    patterns = []
    for current_max in range(1, max_unique_layouts+1):
        pre_sacks = list(combinations(_widths, current_max))
        for widths in pre_sacks:
            # your call to knapsack
            pass
    return patterns
```

Call your function


```python
seed_patterns(_widths, W)
```




    [[Counter({170: 23}), 90],
     [Counter({280: 14}), 80],
     [Counter({320: 12}), 160],
     [Counter({170: 12, 280: 7}), 0],
     [Counter({170: 16, 320: 4}), 0],
     [Counter({280: 12, 320: 2}), 0],
     [Counter({170: 12, 280: 7}), 0]]



For giggles, check the speed of your function using `%%timeit`


```python
%%timeit
patterns = seed_patterns(_widths, W)
```

    69.9 ms ± 5.73 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)


### 1.2.2 More permutations

This is grand, but notice there are additional patterns that may be useful for our stock cutting problem.

We were able to find:

`[Counter({280: 12, 320: 2}), 0],`

but notice how:

`[Counter({320: 9, 280: 4}), 0]`

is also a valid solution to fitting the two slit widths on stock. And in fact, the second solution may be one we need to produce our orders in as few stock rolls as possible. We'll come back to this question later on.

## 1.3 Comparison with old Alg


```python
%%timeit
old_seed_patterns(_widths, [200, 200, 200], W, len(_widths), verbiose=False)
```

    717 ms ± 83.2 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)



```python
import itertools
import numpy as np
```


```python
old_seed_patterns(_widths, [200, 200, 200], W, len(_widths), goal=10, verbiose=False)
```




    [array([Counter({170: 12, 280: 7}), 0], dtype=object),
     array([Counter({170: 16, 320: 4}), 0], dtype=object),
     array([Counter({280: 12, 320: 2}), 0], dtype=object),
     array([Counter({320: 9, 280: 4}), 0], dtype=object),
     array([Counter({170: 23}), 90], dtype=object),
     array([Counter({280: 14}), 80], dtype=object),
     array([Counter({320: 12}), 160], dtype=object)]




```python
def old_seed_patterns(w, q, B, n, max_combinations=3, goal=3, verbiose=True):
    '''
    creates a number of optimal patterns for deckling

    Parameters
    ----------
    w: list
        list of widths (int)
    q: list
        list of rolls for each width (int)
    B: int
        usuable width per mother roll
    n: list
        neck in for each width (int)
    max_combinations: int, default 3
        maximum number of unique products (widths) to have on a mother roll
    goal: int, default 3
        the desired number of recovered patterns from the knapsack problem
        for every unique grouping of unique widths at max_combinations
    verbiose: bool, default True
        turns on/off print statements during execution

    Returns
    -------
    patterns: list of lists
        list of pattern, loss pairs. Pattern is a dictionary containing a width,
        count pair that describes the pattern on the mother roll. Loss is the
        percent material loss for the pattern.
    layout: list
        list of counts for every width on the mother roll. Layout is the best
        possible pattern in terms of minimizing mother rolls to create the order
        with a single pattern.
    '''
    # layout = make_best_pattern(q, w, n, B, verbiose=verbiose)
    combos = []
    for i in range(1,max_combinations+1)[::-1]:
        combos += list(itertools.combinations(w,r=i))
    if verbiose:
        print('')
        print("{} possible max {} combinations".format(len(combos),max_combinations))
    patterns = []
    for combo in combos:
        for combo in list(itertools.permutations(combo)):
            # due to the naive soln, combos of len 1 should be skipped.

            # knapsack/store_patterns will only find one solution (0?) if
            # the width is over half the length of the bin
            if len(combo) == 1:
                sub_goal = 1

            # arbitrary conditional
            elif sum(combo) > (B - min(combo)):
                sub_goal = 1
                # if B / combo[0] < 2:
                #     sub_goal = 1
                # else:
                #     sub_goal = goal
            else:
                sub_goal = goal
            # only provide knapsack with relevant variables
            s = []
            for i in combo:
                s += (int(B/i)*[i])
            t = initt(B,s)
            knapsack(s, B, len(s), t)
            t = np.array(t)
            patterns += store_patterns(t, s, B, goal=sub_goal)
            for j in range(3):
                for i in patterns:
                    for key in list(i[0].keys()):
                        loss = (B - np.sum(np.array(list(i[0].keys())) *
                            np.array(list(i[0].values())))) - key
                        if loss > 0:
                            i[0][key] += 1
                            i[1] = loss
    uni_list = []
    for i in patterns:
        if i not in uni_list:
            uni_list.append(i)
    patterns = uni_list
    patterns = list(np.array(patterns)[np.array(patterns)[:,1]>=0])

    # the naive patterns should be kept due to their usefullness
    # in order fulfilment regardless of loss
    # naive = init_layouts(B, w)
    # for i in naive:
    #     i = [-j for j in i]
    #     patterns.append([dict(zip(w,i)),0])

    if verbiose:
        print("{} unique patterns found".format(len(patterns)))
    return patterns

def store_patterns(t, s, B, goal=5):
    t = np.array(t)
    patterns = []
    bit = 1
    empty = False
    while (len(patterns) < goal):
        found = 0
        for pair in np.argwhere(t == t.flatten()[t.flatten().argsort()[-bit]]):
            N, W = pair
            sack = reconstruct(N, W, t, s)

            # terminate search if we are returning empty sets
            if sack == set():
                empty = True
                break

            pattern = Counter(np.array(s)[list(sack)])
            loss = B - np.sum(np.array(list(pattern.keys())) *
                            np.array(list(pattern.values())))
            if loss >= 0:
                patterns.append([pattern, loss])
            if len(patterns) >= goal:
                break
            found += 1
            if found > 1:
                break
        bit += 1
        if empty:
            break
    return patterns
```
