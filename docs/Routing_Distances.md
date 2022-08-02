<a href="https://colab.research.google.com/github/wesleydatavant/datascience/blob/main/notebooks/Routing_Distances.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Routing Distances

<br>

---

<br>

In this project notebook we'll be comparing for loop and vectorized implemented distance calculations. With large datasets, the vectorized formulation becomes more than 1000 times faster!

<br>

---

## Imports


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist
from haversine import haversine, Unit
from math import isclose
```

## Sample Data


```python
tx_json = {"city":{"4":"Dallas","5":"Houston","33":"Austin","55":"Fort Worth"},
           "state_name":{"4":"Texas","5":"Texas","33":"Texas","55":"Texas"},
           "lat":{"4":32.7935,"5":29.786,"33":30.3005,"55":32.7817},
           "lng":{"4":-96.7667,"5":-95.3885,"33":-97.7522,"55":-97.3474}}
df = pd.DataFrame(tx_json)
lat = df['lat'].values
lng = df['lng'].values
x = np.array([lat, lng]).T
y = x.copy()
```

### Pseudo Large Dataset for Speed Check


```python
a = np.tile(x.T,1000).T
b = np.tile(y.T,1000).T
print(a.shape)
print(b.shape)
```

    (4000, 2)
    (4000, 2)


## Euclidean Distance


```python
def dist(a, b):
    
    """
    returns the geometric distance of two long lat coordinates
    
    Parameters
    ----------
    a : numpy array
        the long lat coordinate of the patient
    b : numpy array
        the long lat coordinate of the site(s). can either be a single
    site or multiple sites
    
    Returns
    -------
    distances : array or float
        dist will return a float if only a single site is specified in b,
        otherwise will return an array
    """
    
    # if a is a list of patient zips
    if len(a.shape) > 1:
        a = a[:, None, :]
        b = b[None, :, :]
        
    return np.sqrt(((a-b)**2).sum(axis=-1))
      
    # else if b is a list of site zips
    if len(b.shape) > 1:
        b = b.T
        
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def dist_no_np(a_, b_):
    res = []
    for a in a_:
        ress = []
        for b in b_:
            ress.append(np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2))
        res.append(ress)
```

### Speed Check


```python
%%timeit
dist(a, b)
```

    473 µs ± 19.1 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)



```python
%%timeit
dist_no_np(a, b)
```

    161 ms ± 457 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)



```python
%%timeit
cdist(a, b, metric='euclidean')
```

    126 µs ± 482 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)



```python
euclid_result = dist(x, y)
```


```python
cdist(x, y, metric='euclidean')
```




    array([[0.        , 3.30824598, 2.68071991, 0.58081988],
           [3.30824598, 0.        , 2.41904691, 3.57931665],
           [2.68071991, 2.41904691, 0.        , 2.51400407],
           [0.58081988, 3.57931665, 2.51400407, 0.        ]])



## Haversine Distance

### Non Vectorized


```python
# if you have the local file https://simplemaps.com/data/us-cities
df = pd.read_csv("../../Downloads/simplemaps_uscities_basicv1.75/uscities.csv")
df = df.loc[(df['city'].isin(['Austin', 'Dallas', 'Fort Worth', 'Houston'])) &
       (df['state_id'] == 'TX')]

# df[['city', 'state_name', 'lat', 'lng']].to_json()
```


```python
def test_haversine():
    tx_json = {"city":{"4":"Dallas","5":"Houston","33":"Austin","55":"Fort Worth"},
               "state_name":{"4":"Texas","5":"Texas","33":"Texas","55":"Texas"},
               "lat":{"4":32.7935,"5":29.786,"33":30.3005,"55":32.7817},
               "lng":{"4":-96.7667,"5":-95.3885,"33":-97.7522,"55":-97.3474}}
    df = pd.DataFrame(tx_json)
    lat = df['lat'].values
    lng = df['lng'].values
    x = np.array([lat, lng]).T
    y = x.copy()
    # https://stackoverflow.com/questions/58307211/python-generate-distance-matrix-for-large-number-of-locations
    answer = cdist(x, y, metric=haversine, unit= Unit.MILES)
    tx_distances = np.array([[  0.        , 223.15619132, 181.75853929,  33.74018707],
                           [223.15619132,   0.        , 145.77171262, 237.09579454],
                           [181.75853929, 145.77171262,   0.        , 173.08331265],
                           [ 33.74018707, 237.09579454, 173.08331265,   0.        ]])
    isvect = np.vectorize(isclose)
    assert isvect(answer, tx_distances, abs_tol=1).all(), "distances are incorrect"
    print("Test Passed")
```


```python
test_haversine()
```

    Test Passed



```python
EARTH_RADIUS = 3958.76
# Copied from the pgeocode library (along with the zip code lat/lon values)
# https://github.com/symerio/pgeocode/blob/96b86f5e8b9569de0dd5eaa37385cb06752bb944/pgeocode.py#L396
# note that the docstring is incorrect as this will only work with arrays of shape (1, 2)
def haversine_distance(x, y):
    """Haversine (great circle) distance
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    Parameters
    ----------
    x : array, shape=(n_samples, 2)
      the first list of coordinates (degrees)
    y : array: shape=(n_samples, 2)
      the second list of coordinates (degress)
    Returns
    -------
    d : array, shape=(n_samples,)
      the distance between corrdinates (km)
    References
    ----------
    https://en.wikipedia.org/wiki/Great-circle_distance
    """
    # if the points are the same, their distance is zero
    if x[0] == y[0] and x[1] == y[1]:
        return 0
    x_rad = np.radians(x)
    y_rad = np.radians(y)

    
    d = y_rad - x_rad

    dlat, dlon = d.T
    x_lat = x_rad[0]
    y_lat = y_rad[0]

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(x_lat) * np.cos(y_lat) * np.sin(dlon / 2.0) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS * c

def hav_dist_no_np(a_, b_):
    res = []
    for a in a_:
        ress = []
        for b in b_:
            ress.append(haversine_distance(a, b))
        res.append(ress)
    return res
```


```python
def test_haversine_non_vect():
    tx_json = {"city":{"4":"Dallas","5":"Houston","33":"Austin","55":"Fort Worth"},
               "state_name":{"4":"Texas","5":"Texas","33":"Texas","55":"Texas"},
               "lat":{"4":32.7935,"5":29.786,"33":30.3005,"55":32.7817},
               "lng":{"4":-96.7667,"5":-95.3885,"33":-97.7522,"55":-97.3474}}
    df = pd.DataFrame(tx_json)
    lat = df['lat'].values
    lng = df['lng'].values
    x = np.array([lat, lng]).T
    y = x.copy()
    answer = np.array(hav_dist_no_np(x, y))
    tx_distances = np.array([[  0.        , 223.15619132, 181.75853929,  33.74018707],
                           [223.15619132,   0.        , 145.77171262, 237.09579454],
                           [181.75853929, 145.77171262,   0.        , 173.08331265],
                           [ 33.74018707, 237.09579454, 173.08331265,   0.        ]])
    isvect = np.vectorize(isclose)
    assert isvect(answer, tx_distances, abs_tol=1).all(), "distances are incorrect"
    print("Test Passed")
```


```python
test_haversine_non_vect()
```

    Test Passed


### Vectorized


```python
def haversine_vectorized(x, y):
    """
    Vectorized haversine formual will work with arrays of long/lat coordinates
    
    Parameters
    ----------
    x : array, shape=(n_samples_1, 2)
      the first list of coordinates (degrees)
    y : array: shape=(n_samples_2, 2)
      the second list of coordinates (degress)
    Returns
    -------
    d : array, shape=(n_samples_1, n_samples_2)
      the distance between corrdinates (km)
    """
    EARTH_RADIUS = 3958.76
    
    # convert all latitudes/longitudes from decimal degrees to radians
    x_rad = np.radians(x)
    y_rad = np.radians(y)

    # unpack latitude/longitude
    x_lat, x_lng = x_rad[:, 0], x_rad[:, 1]
    y_lat, y_lng = y_rad[:, 0], y_rad[:, 1]

    # broadcast, can also use np.expand_dims
    x_lat = x_lat[None, :]
    x_lng = x_lng[None, :]
    y_lat = y_lat[:, None]
    y_lng = y_lng[:, None]

    # calculate haversine
    d_lat = y_lat - x_lat
    d_lng = y_lng - x_lng

    a = (np.sin(d_lat / 2.0) ** 2
         + np.cos(x_lat) * np.cos(y_lat) * np.sin(d_lng / 2.0) ** 2)

    return 2 * EARTH_RADIUS * np.arcsin(np.sqrt(a))
```


```python
haversine_vectorized(x, y)
```




    array([[  0.        , 223.15611622, 181.75847812,  33.74017572],
           [223.15611622,   0.        , 145.77166356, 237.09571474],
           [181.75847812, 145.77166356,   0.        , 173.0832544 ],
           [ 33.74017572, 237.09571474, 173.0832544 ,   0.        ]])




```python
def test_haversine_vect():
    tx_json = {"city":{"4":"Dallas","5":"Houston","33":"Austin","55":"Fort Worth"},
               "state_name":{"4":"Texas","5":"Texas","33":"Texas","55":"Texas"},
               "lat":{"4":32.7935,"5":29.786,"33":30.3005,"55":32.7817},
               "lng":{"4":-96.7667,"5":-95.3885,"33":-97.7522,"55":-97.3474}}
    df = pd.DataFrame(tx_json)
    lat = df['lat'].values
    lng = df['lng'].values
    x = np.array([lat, lng]).T
    y = x.copy()
    answer = haversine_vectorized(x, y)
    tx_distances = np.array([[  0.        , 223.15619132, 181.75853929,  33.74018707],
                           [223.15619132,   0.        , 145.77171262, 237.09579454],
                           [181.75853929, 145.77171262,   0.        , 173.08331265],
                           [ 33.74018707, 237.09579454, 173.08331265,   0.        ]])
    isvect = np.vectorize(isclose)
    assert isvect(answer, tx_distances, abs_tol=1).all(), "distances are incorrect"
    print("Test Passed")
```


```python
test_haversine_vect()
```

    Test Passed


### Speed Check

Test with small arrays leads to 10x speedup


```python
%%timeit
haversine_vectorized(x, y)
```

    7.86 µs ± 42 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)



```python
%%timeit
hav_dist_no_np(x, y)
```

    62.4 µs ± 434 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)


Test with large arrays leads to >1000x speedup


```python
%%timeit
haversine_vectorized(np.tile(x.T,1000).T, np.tile(x.T,1000).T)
```

    277 ms ± 1.65 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)



```python
%%timeit
hav_dist_no_np(np.tile(x.T,1000).T, np.tile(x.T,1000).T)
```

    1min 1s ± 285 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

