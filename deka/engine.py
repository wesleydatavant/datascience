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