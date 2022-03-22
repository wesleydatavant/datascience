import numpy as np
from collections import Counter

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
        
def initt(W, val):
    return [[None for i in range(W + 1)] for j in range(len(val) + 1)]

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