<a href="https://colab.research.google.com/github/wesleybeckner/deka/blob/main/notebooks/solutions/SOLN_P1_Stock_Cutting.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Stock Cutting Part 1:<br> Finding Good Patterns

<br>

---

<br>

In this project notebook we'll be laying the foundations of stock cutting. We'll begin by discussing the common dynamic programming problem: the knapsack problem

<br>

---

## 1.0: What is the knapsack Problem?

<p align=center>
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/Knapsack.svg/500px-Knapsack.svg.png"></img>
</p>

**From Wikipedia**:

The knapsack problem is a problem in combinatorial optimization: Given a set of items, each with a weight and a value, determine the number of each item to include in a collection so that the total weight is less than or equal to a given limit and the total value is as large as possible. It derives its name from the problem faced by someone who is constrained by a fixed-size knapsack and must fill it with the most valuable items. The problem often arises in resource allocation where the decision makers have to choose from a set of non-divisible projects or tasks under a fixed budget or time constraint, respectively.

## 1.2: Simple Knapsack

Say we have the following items we want to fit into a bag. Our total weight limit is 8 lbs. The items are worth 1, 2, 5, and 6 dollars apiece and weigh 2, 3, 4, and 5 lbs.

```
val = [1, 2, 5, 6]
wt = [2, 3, 4, 5]
W = 8
```
We may be able to solve in our own minds that the maximum value here is items of value 2 and 6 for a total value of 8 dollars and total weight of 8 lbs. How do we solve this algorithmically?


```python
val = [1, 2, 5, 6]
wt = [2, 3, 4, 5]
W = 8
```

the table we would fill out looks like the following

| Value | Weight | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|-------|--------|---|---|---|---|---|---|---|---|---|
| None  | None   | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1     | 2      | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 2     | 3      | 0 | 0 | 1 | 2 | 2 | 3 | 3 | 3 | 3 |
| 5     | 4      | 0 | 0 | 1 | 2 | 5 | 5 | 6 | 6 | 7 |
| 6     | 5      | 0 | 0 | 1 | 2 | 5 | 6 | 6 | 7 | 8 |

The column headers (after Value and Weight) indicate the maximum _weight_ allowable under that column. Every row indicates what _items_ are allowable i.e. on row 2 we only consider item 1, on row 3 we consider items 1 _and_ 2, and so forth. 

Let's walk through a few cells.

1. In the first two columns, we can't include any items under this weight restriction, hence all cell values are 0. 
2. In column 2, row 2, we are able to fit item 1 under this weight constraint, so the value of our knapsack increases to 1
3. In the remaining cells of row 2, our value is persistant at a total of 1, since we are only considering the first item in this row
4. Moving onto the second row, things get interesting at row 3, column 3
    1. if n is our current item, and w is the current weight (i.e. n is the row and w is the column)
    2. then we want to fill in the current cell `t[n,w]` with the maximum value: `t[n-1,w]` or `t[n-1, W-wt[n]] + val[n]`

Let's make sense of the `max` statement in 4.B. We know that at the current cell, we can always default to the value of the cell above (`t[n-1,w]`). The trick comes in the second part of the `max` statement. We look at the total value if we were to add the current item (`val[n]`), to the value back in the table in the previous row `[n-1]` where the total weight is equal to the current weight under consideration minus the weight of the current item (`W-wt[n]`)

Now we'll go into our second point. Take a look at row 4, column 3 (directly below the cell we were considering before) in this case, the weight of the current item (4 lbs) is more than the maximum allowable weight (3 lbs) and so in this case we default to the value in the cell above (`t[n-1, w]`)

algorithmically this looks like the following:

```
if wt[n] <= W:
        return max(t[n-1,w], t[n-1, W-wt[n]] + val[n])
    
elif wt[n] > W:
    return t[n-1, w]
```

### 1.2.1 Use recursion to find the optimum value of the knapsack

The insight we require is that in the above code block, we want to run this conditional on the bottom right corner of the table, where we have allowed the maximum possible weight in the bag and included all items for consideration. But in order to ask this we need to know the values in the preceding cells of the table! This is where _recurssion_ comes in handy. Our recursive algorithm will return `t[n][w]` for the given parameters of `wt` (weights), `val` (values), `W` (max allowable weight), and, well `t`, the table itself.

#### 🎒 Exercise 1: Create the empty table as a list of lists

Before writing the knapsack function, we will need to initialize our empty table:


```python
# We initialize the table. note that I am adding a 0th row and a 0th column
# at the top and left of the table respectively. The table
# needs to be len(val) + 1 rows and W + 1 columns
t = [[None for i in range(W + 1)] for j in range(len(val) + 1)]
t
```




    [[None, None, None, None, None, None, None, None, None],
     [None, None, None, None, None, None, None, None, None],
     [None, None, None, None, None, None, None, None, None],
     [None, None, None, None, None, None, None, None, None],
     [None, None, None, None, None, None, None, None, None]]



after you've propery coded out the table wrap this into a function `initt`


```python
# we'll wrap the above into a function for ease of use
def initt(W, val):
    return [[None for i in range(W + 1)] for j in range(len(val) + 1)]
```

#### 🎒 Exercise 2: Complete the function `knapsack` using recursion without memoization

Note that we need to think about how our indexing will work between the table, `t`, and our list representations of our items, `wt` and `val`. index 0 in our table means _no items_ and _no weight_ whereas in our list index 0 reprents _item 1_ so in the following knapsack function we need to shift all our indexing's of `wt` and `val` backwards by 1 (this is the difference between the code in the function and the mathematical representation in Section 1.2)


```python
def knapsack(wt, val, w, n, t):
    # n, w will be the row, column of our table
    # solve the basecase. 
    if w == 0 or n == 0:
        return 0
    
    # now include the conditionals
    if wt[n-1] <= w:
        t[n][w] = max(
            knapsack(wt, val, w, n-1, t),
            knapsack(wt, val, w-wt[n-1], n-1, t) + val[n-1])
        return t[n][w]
    
    elif wt[n-1] > w:
        t[n][w] = knapsack(wt, val, w, n-1, t)
        return t[n][w]
```


```python
knapsack(wt, val, W, len(val), t)
```




    8



If we look at the output of our table, we will see that it matches the one we described in Section 1.2

| Value | Weight | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|-------|--------|---|---|---|---|---|---|---|---|---|
| None  | None   | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1     | 2      | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 2     | 3      | 0 | 0 | 1 | 2 | 2 | 3 | 3 | 3 | 3 |
| 5     | 4      | 0 | 0 | 1 | 2 | 5 | 5 | 6 | 6 | 7 |
| 6     | 5      | 0 | 0 | 1 | 2 | 5 | 6 | 6 | 7 | 8 |

it turns out that we did not have to visit every cell in the table however, to arrive at the optimal value of our knapsack:


```python
t
```




    [[None, None, None, None, None, None, None, None, None],
     [None, 0, None, 1, 1, 1, None, None, 1],
     [None, None, None, 2, 2, None, None, None, 3],
     [None, None, None, 2, None, None, None, None, 7],
     [None, None, None, None, None, None, None, None, 8]]



### 1.2.2 Enrichment: use memoization to speed up the solution

Memoization is a techniqued used along with recursion to create _dynamic programms_. We let our recursive algorithm know when we've already visited a subproblem by passing along a table (which in this case, we already have)

#### 🎒 Exercise 3: Check if we have already visited a location in the table


```python
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
```

### 1.2.3 Reconstruct what items were added to the knapsack by reading the table

The next part of the puzzle is to determine what items were added to the knapsack to achieve this optimal result! We can infer this from the table. Take a look at the table again, the algorithm we will envoke is that if the current cell's value does not appear in the previous row, it is because added that row's item! To see what else is in that hypothetical knapsack, we simply move up a row and left the number of columns equal to the last item we added, and repeat the process

#### 🎒 Exercise 4: Complete the pseudo code below to create a `set` that includes all the items in the knapsack


```python
recon = set()
column = W
# we will iterate through the rows of the table from bottom to top
for row in range(len(val))[::-1]:
    # we know that if the current cell value is not anywhere in the
    # previous row, it is because we added that item to the knapsack
    # remember that the table indexing is shifted forward compared to
    # the list indexes
    if t[row+1][column] not in t[row]:
        recon.add(row)
        # after we add the item, we need to adjust the weight the appropriate
        # number of steps
        column -= wt[row]
        
        # we will stop after we reach the 0th (no weight) column
        if column < 0:
            break
        else:
            continue
recon
```




    {1, 3}



#### 🎒 Exercise 5: Wrap this in a function called `reconstruct`

we see that the output of the reconstruction are the second and fourth items (index starts at 0), which is what we expected! Let's package this into a function


```python
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
```

Now, to translate these indices to actual items we're going to use a function called _Counter_

> Dict subclass for counting hashable items.  Sometimes called a bag
or multiset.  Elements are stored as dictionary keys and their counts
are stored as dictionary values.

We'll use Counter along with Numpy to take our indices/val list and count the number of times an item was used in the solution


```python
import numpy as np
from collections import Counter
```


```python
sack = reconstruct(len(val), W, t, wt)
pattern = Counter(np.array(val)[list(sack)])
pattern
```




    Counter({2: 1, 6: 1})



## 1.3 Unit Tests

It is always good practice to test as we go. Let's make sure our algorithm works on some known subproblems


```python
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
```


```python
test_small_bag()
```

    Optimal value found
    Optimal items found



```python
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
test_val_weight_equality()
```

    Optimal value found
    Optimal items found


## 1.4 Save our functions


```python
import numpy as np
from collections import Counter

def knapsack(wt, val, W, n, t):
    """
    in our particular case, the wt and val of the item are the same
    wt/val is 's' and is the items sorted by length, increasing
    wt: list
        the ordered weights (lengths of the rolls)
    val: list
        the ordered values (lengths of the rolls)
    W: int
        the weight of the current knapsack (the used length of the mother roll)
    n: int
        the number of items in the mother roll
    t: list of list
        the knapsack table
    """

    # base conditions
    if n == 0 or W == 0:
        # the first row and first column of the table is filled with 0s
        return 0
    if t[n][W] != -1: 
        # already solved at [n][W]
        return t[n][W]

    # choice diagram code
    if wt[n-1] <= W:
        # if the previous item under consideration is less than the current
        # weight left in the mother roll
        
        # we can now do 1 of 2 things add the new item in (and take the val
        # of the bag at the previous lvl w/o the item) (the answer to the left
        # in the table) plus the new wt/val or we use the best answer from one
        # fewer items at this weight lvl (the answer above the current cell in
        # the table)
        # t[n,w] = max{t[n-1,w], t[n-1, w-w[n-1]] + val[n-1]}
        # note that in the following wt and val are indexed starting at 0
        # but t/knapsack is indexed starting at 1 (index 0 in the table is all
        # 0's)
        t[n][W] = max(
            val[n-1] + knapsack(
            wt, val, W-wt[n-1], n-1, t),
            knapsack(wt, val, W, n-1, t))
        
        return t[n][W]
    elif wt[n-1] > W:
        # if wt/val of the current item under consideration is more than the
        # weight left in the bag, we've already found the best solution for 
        # this number of items
        t[n][W] = knapsack(wt, val, W, n-1, t)
        return t[n][W]

def initt(B, lens):
    """
    t, the returned table, will be a list of lists, but if transformed to 
    an array it takes the shape of the number of products + 1 (len(s) + 1) 
    by the usable width + 1 (B + 1)
    """
    # We initialize the matrix with -1 at first.
    return [[-1 for i in range(B + 1)] for j in range(lens + 1)]

def reconstruct(N, W, t, wt):
    recon = set()
    for j in range(N)[::-1]:
        if t[j+1,W] not in t[j,:]:
            recon.add(j)
            W = W - wt[j] # move columns in table lookup
        if W < 0:
            break
        else:
            continue
    return recon
```

## 1.5 Lab

In the deka repository add our functions to `enginge.py` and our unit_tests to `test_engine.py` make sure you can run the tests via the Makefile
