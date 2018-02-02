from numba import jit
import numpy as np


@jit
def select_closest(candidates: int, origin: int) -> int:
    """
    Return the index of the closest candidate to a given point.
    """
    return euclidean_distance(candidates, origin).argmin()


@jit
def euclidean_distance(a: int, b: int) -> 'numpy.ndarray':
    """
    Return the array of distances of two numpy arrays of points.
    """
    return np.linalg.norm(a - b, axis=1)


@jit
def route_distance(cities: 'pandas.DataFrame') -> float:
    """
    Return the cost of traversing a route of cities in a certain order.
    """
    points = cities[['x', 'y']]
    distances = euclidean_distance(points, np.roll(points, 1, axis=0))
    return np.sum(distances)
