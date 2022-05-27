<a href="https://colab.research.google.com/github/wesleybeckner/deka/blob/main/notebooks/solutions/SOLN_P6_Stock_Cutting.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Stock Cutting Part 6:<br> API

<br>

---

<br>

In this project notebook we'll be showcasing how the API connects to the deka logic

<br>

---

## Import Functions and Libraries

Note: in this notebook I have switched the variable `W` to `B`, this represents the width of the mother roll


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

def pack_knap(wt, val, B):
    new_wt = []
    new_val = []
    for w, v in zip(wt, val):
        new_wt += [w]*int(B/w)
        new_val += [v]*int(B/w)
    wt = new_wt
    val = new_val
    t = initt(B, val)
    best = knapsack(wt, val, B, len(val), t)
    loss = B - best
    sack = reconstruct(len(val), B, t, wt)
    pattern = Counter([wt[i] for i in list(sack)])
    
    value = Counter([val[i] for i in list(sack)])
    

    total = 0
    for worth, multiple in value.items():
        total += worth * multiple
    return pattern, total

def seed_patterns(_widths, B, max_widths=3):
    patterns = []
    for current_max in range(1, max_widths+1):
        pre_sacks = list(combinations(_widths, current_max))
        for widths in pre_sacks:
            new = []
            for w in widths:
                new += [w]*int(B/w)
            widths = new

            t = initt(B, widths)
            best = knapsack(widths, widths, B, len(widths), t)
            loss = B - best
            sack = reconstruct(len(widths), B, t, widths)
            pattern = Counter([widths[i] for i in list(sack)])
            if [pattern, loss] not in patterns:
                patterns.append([pattern, loss])
    return patterns

def initt(B, val):
    return [[None for i in range(B + 1)] for j in range(len(val) + 1)]

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
    
def reconstruct(N, B, t, wt):
    recon = set()
    for j in range(N)[::-1]:
        if (t[j+1][B] not in t[j]) and (t[j+1][B] != 0):
            recon.add(j)
            B = B - wt[j] # move columns in table lookup
        if B < 0:
            break
        else:
            continue
    return recon

def test_small_bag():
    # the problem parameters
    val = [60, 50, 70, 30]
    wt = [5, 3, 4, 2]
    B = 5

    # the known solution
    max_val = 80
    max_items = [50, 30]

    t = initt(B, val)
    best = knapsack(wt, val, B, len(val), t)
    sack = reconstruct(len(val), B, t, wt)
    pattern = Counter([val[i] for i in list(sack)])

    assert best == max_val, "Optimal value not found"
    print("Optimal value found")

    assert list(pattern.keys()) == max_items, "Optimal items not found"
    print("Optimal items found")
    
def test_val_weight_equality():
    # the problem parameters
    val = wt = [2, 2, 2, 2, 5, 5, 5, 5]
    B = 14

    # the known solution
    max_val = 14
    max_items = Counter([5, 5, 2, 2])

    t = initt(B, val)
    best = knapsack(wt, val, B, len(val), t)
    sack = reconstruct(len(val), B, t, wt)
    pattern = Counter([val[i] for i in list(sack)])

    assert best == max_val, "Optimal value not found"
    print("Optimal value found")

    assert pattern == max_items, "Optimal items not found"
    print("Optimal items found")

def test_simple_stock_cutting():
    q = [80, 50, 100]
    widths = w = [4, 6, 7]
    B = 15
    ans = 96

    patterns = seed_patterns(widths, B, max_widths=1)

    while True:
        X, val = solveX(patterns, widths, q)
        pattern, total = pack_knap(w, val, B)
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
    B = 4160
    ans = 59

    patterns = seed_patterns(widths, B, max_widths=1)

    while True:
        X, val = solveX(patterns, widths, q)
        pattern, total = pack_knap(widths, val, B)
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


## Defining Objects


```python
# from test_stock_cutting_2
w = [170, 234, 158]
n = [2, 2, 2]
widths = [i+j for i, j in zip (n, w)]
q = [879, 244, 181]
B = 4160
ans = 59

# a new variable that we have not yet defined will be the length of the mother rolls
L = 17000

# we will pass our deckle data around in an object called a bucket
bucket = dict()
bucket['w'] = w
bucket['n'] = n
bucket['L'] = L
bucket['q'] = q
bucket['B'] = B

display(bucket)
```


    {'w': [170, 234, 158],
     'n': [2, 2, 2],
     'L': 17000,
     'q': [879, 244, 181],
     'B': 4160}


In addition to defining a bucket object, we will also formalize the hyperparameters for our deckle algorithm

* Primary:
    * `bucket`: dict, the collection of deckle data
        * `w`: list of ints, slit widths
        * `n`: list of ints, knife in loss
        * `L`: int, length of the mother roll
        * `q`: list of ints, quantities ordered of each slit width
        * `B`: int, the width of the mother roll
    * `max_widths`: int, the maximum number of unique widths we want on any specific layout
    * `max_layouts`: int, the maximum number of unique layouts we want to fullfill the deckle solution


```python
def deckle(bucket, max_widths=3, max_layouts=3):
    w = bucket['w']
    n = bucket['n']
    L = bucket['L']
    q = bucket['q']
    B = bucket['B']
    widths = [i+j for i, j in zip (n, w)]
    
    solutions = []
    
    # make the patterns first (pre_sacks)
    # current_max will account for all max_layouts and anything less
    for current_max in range(1, max_layouts+1):

        # our pre_sacks contains the collection of combinations
        pre_sacks = list(combinations(patterns, current_max))

        # the sack is what we will send to the lin prog program
        # we will borrow from column_gen to supplement any pattern 
        # combinations that could use an extra layout (to get to
        # the max_layouts value)
        # for sack in pre_sacks:
        #     print(sack)
        # print()
    
    # current_max will account for all max_layouts and anything less
    for current_max in range(1, max_layouts+1):

        # our pre_sacks contains the collection of combinations
        pre_sacks = list(combinations(patterns, current_max))

        # the sack is what we will send to the lin prog program
        # we will borrow from column_gen to supplement any pattern 
        # combinations that could use an extra layout (to get to
        # the max_layouts value)
        for sack in pre_sacks:
            sack = list(sack)
            while True:
                X, val = solveX(sack, widths, q)
                pattern, total = pack_knap(widths, val, B)
                if (total > 1) and (len(sack) < max_layouts):
                    sack.append([pattern, 0])
                    continue
                break
            if sum(X) > 0:
                # print(total)
                # print(len(sack))
                # print(f"total doffs: {sum(X)}")
                for quant, pattern in zip(X, sack):
                    if quant > 0:
                        print(f"{quant}, {pattern[0]}")
                print()
                solutions.append([X, sack])
    return solutions
```


```python

```

## Parallel Processing

For our parallel processing wrapper we introduce a couple more hyperparameters that will allow us to seach through a broader optimization space

* Secondary:
    * `production_targets`: list of floats, default `[1]`. Fraction of each quantity in `q` to be passed to optimization task
    * `edge_trim_allowance`: list of ints, default `[0]`. The amount of edge trim (mm) to _add_ to each end of the mother roll. 


```python
import math
import multiprocessing as mp
```


```python
def make_search(bucket, max_widths, max_layouts, production_targets=[1],
    edge_trim_allowance=[0], verbose=False):
    """
    Handles the parallel search for both schedule_api and front_search_api
    """

    params = []
    goals = [1, 3, 10, 100, 1000]
    
    # min_patterns will tell us if we can fit all widths into a single mother roll or not
    min_patterns = math.ceil(sum(bucket['w']) / bucket['B'])
    
    # we will mix n match every possible deckle request that comes from the hyperparameters
    # i.e. max 1 width/layout + max 3 layouts + 99% production + 10mm edge allowance
    for current_max_width in range(max_widths+1):
        for current_max_layout in range(min_patterns, max_layouts+1):
            for target in production_targets:
                for edge in edge_trim_allowance:
                    
                    # only keep combinations that are possible to solve ie the product
                    # of max_layouts and max_widths needs to be greater than the number
                    # of unique widths (bucket['w'])
                    for recursion_goal in goals[:goal]:
                        if current_max_width * current_max_layout >= len(bucket['w']):
                            
                            # the params list will be used by multiprocessing library to find 
                            # each request
                            params.append([bucket, current_max_width, current_max_layout, target,
                                          edge, recursion_goal, verbiose, opt])
    return params
```


```python
params = make_search(bucket, 3, 3)
```


```python
len(params)
```




    18



We now want to solve the deckle problem 18 times


```python
pool = mp.Pool(mp.cpu_count())
results = [pool.apply_async(deckle, args=(param[0],
                                    param[1],
                                    param[2],
                                    param[3],
                                    param[4],
                                    param[5],
                                    param[6],
                                    param[7])) for param in params]
pool.close()
while True:
    if all([i.ready() for i in results]):
        res = []
        for i in results:
            try:
                a_result = i.get().copy()
                a_result[3]['loss'] = round(a_result[3]['loss'].astype(float), 3)
                res.append(a_result)
            except:
                pass # search had failed

        df = pd.DataFrame(res, columns=['loss', 'jumbos', 'inventory',
            'summary', 'combinations', 'patterns', 'target', 'edge'])

        df['str summ'] = df['summary'].astype(str)
        df = df.sort_values('loss').reset_index(drop=True)
        df = df.drop_duplicates(['str summ'])[['loss', 'jumbos',
            'inventory', 'summary', 'combinations', 'patterns', 'target',
            'edge']].reset_index(drop=True)
        df['loss rank'] = df['loss'].rank(method='first')
        df['jumbo rank'] = df['jumbos'].rank(method='first')
        break
df['loss'] = round(df['loss'], 2)
df = df.sort_values('jumbo rank').reset_index(drop=True)
```

## What we get from the front end


```python
req = {
                "width1": "818",
                "width2": "1638",
                "width3": "",
                "width4": "",
                "width5": "",
                "width6": "",
                "roll1": "473",
                "roll2": "241",
                "roll3": "",
                "roll4": "",
                "roll5": "",
                "roll6": "",
                "neck1": "6",
                "neck2": "8",
                "neck3": "",
                "neck4": "",
                "neck5": "",
                "neck6": "",
                "usable_width": "4160",
                "put_up": "11700",
                "doffs_per_jumbo": "1",
                "max_widths": "2",
                "max_layouts": "2",
                "production_targets": "1",
                "min_prod": "0.99",
                "max_prod": "1",
                "edge_allowance": "0"
              }
```

how are this json used to populate the appropriate fields for the deka function calls; and how are the results formatted and sent back to the front end


```python
def standalone(req):
    """
    Translates the request body into inputs for the cutting stock algorithm
    """
    max_layouts = int(req["max_layouts"])
    max_widths = int(req["max_widths"])
    min_prod = float(req["min_prod"])
    max_prod = float(req["max_prod"])
    
    # search between min and max in 1% increments
    if min_prod != max_prod:
        production_targets = list(np.arange(min_prod,max_prod,.01))
    else:
        production_targets = [min_prod]
    
    # search with additional edge trim in 1mm increments
    edge_trim_allowance = int(req["edge_allowance"])
    edge_trim_allowance = list(range(edge_trim_allowance+1))
    
    # build the deckle bucket
    w = [int(req[i]) for i in req.keys() if
         ('width' in i) & (req[i] != '') & ('_width' not in i)]
    q = [int(req[i]) for i in req.keys() if ('roll' in i) & (req[i] != '')]
    q = [math.ceil(i / int(req['doffs_per_jumbo'])) for i in q]
    n = [int(req[i]) for i in req.keys() if ('neck' in i) & (req[i] != '')]
    B = int(req['usable_width'])
    L = int(req['put_up']) * int(req['doffs_per_jumbo'])
    w = [i + j for i, j in zip(w, n)]
    bucket = dict()
    bucket['w'] = w
    bucket['n'] = n
    bucket['L'] = L
    bucket['q'] = q
    bucket['B'] = B
```


```python
def front_search_api(req, version, suppress_max_inv=40): # rename this to standalone
    if version == 1:
        max_layouts = req["max_layouts"].split(', ')
        max_layouts = [int(i) for i in max_layouts]
        max_widths = req["max_widths"].split(', ')
        max_widths = [int(i) for i in max_widths]
        production_targets = req["production_targets"].split(', ')
        production_targets = [float(i) for i in production_targets]
        edge_trim_allowance = req["edge_allowance"].split(', ')
        edge_trim_allowance = [int(i) for i in edge_trim_allowance]
    elif version == 2:
        max_layouts = int(req["max_layouts"])
        max_widths = int(req["max_widths"])
        min_prod = float(req["min_prod"])
        max_prod = float(req["max_prod"])
        if min_prod != max_prod:
            production_targets = list(np.arange(min_prod,max_prod,.01))
        else:
            production_targets = [min_prod]
        edge_trim_allowance = int(req["edge_allowance"])
        edge_trim_allowance = list(range(edge_trim_allowance+1))

    w = [int(req[i]) for i in req.keys() if
         ('width' in i) & (req[i] != '') & ('_width' not in i)]
    q = [int(req[i]) for i in req.keys() if ('roll' in i) & (req[i] != '')]
    q = [math.ceil(i / int(req['doffs_per_jumbo'])) for i in q]
    n = [int(req[i]) for i in req.keys() if ('neck' in i) & (req[i] != '')]
    B = int(req['usable_width'])
    L = int(req['put_up']) * int(req['doffs_per_jumbo'])
    w = [i + j for i, j in zip(w, n)]
    bucket = dict()
    bucket['w'] = w
    bucket['n'] = n
    bucket['L'] = L
    bucket['q'] = q
    bucket['B'] = B

    df = make_search(bucket, max_widths, max_layouts, production_targets,
        edge_trim_allowance, goal=3, opt='time', verbiose=False)

    # compute string formatted layouts, inventory levels, and whether targets
    # were met (%)
    for bucket_ind in range(df.shape[0]):
        tot_q = 0
        for index, layout in enumerate(df.iloc[bucket_ind]
                ['summary']['layout']):
            txt = ''
            for width in layout:
                if layout[width] != 0:
                    txt += str(layout[width]) + 'x' + str(width) + ' + '
                    tot_q += layout[width] * df.iloc[bucket_ind]['summary']['jumbos'][index]
            txt = txt[:-3]
            df.loc[bucket_ind, 'summary'].loc[index, 'layout'] = txt
        # print(tot_q)
        txt = ''
        tot_inv = 0
        for index, layout in enumerate(df.iloc[bucket_ind]['inventory']):
            inventory = df.iloc[bucket_ind]['inventory'][layout]
            df.loc[bucket_ind, 'inv. {}'.format(layout)] = inventory
            txt += str(int(inventory)) + 'x' + str(layout) + ' + '
            tot_inv += inventory
        txt = txt[:-3]
        target = f"{tot_q / (tot_q-tot_inv) * 100:.1f}%"
        df.loc[bucket_ind, 'inventory'] = txt
        df.loc[bucket_ind, 'target'] = target
    if suppress_max_inv:
        df = df.loc[(df[[col for col in df.columns if 'inv.' in col]] < suppress_max_inv).all(1)]
    dfjson = df.to_json(orient="records")
    return dfjson
```
