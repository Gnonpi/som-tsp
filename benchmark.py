import json
import time

import numpy as np
from collections import defaultdict

from src.main import som
from src.io_helper import read_tsp

TSP_ASSETS = [
    'fi10639.tsp'
    'it16862.tsp'
    'qa194.tsp'
    'uy734.tsp'
]
ITERATION_RANGE = np.logspace(2, 4, num=10, dtype=int)
NB_REPETITION = 3


def time_one_solution(filepath, iterations):
    start_time = time.clock()
    df = read_tsp(filepath)
    som(df, iterations, plot=False)
    total_time = time.clock() - start_time
    return total_time


def time_all_iterations():
    times = defaultdict(defaultdict(list))
    for tsp_filepath in TSP_ASSETS:
        for nb_iteration in ITERATION_RANGE:
            for _ in range(NB_REPETITION):
                _time = time_one_solution(tsp_filepath, nb_iteration)
                times[tsp_filepath][nb_iteration].append(_time)
    return times


if __name__ == "__main__":
    dict_times = time_all_iterations()
    with open('time_results.json', 'w') as f:
        json.dump(dict_times, f)
