import numpy as np

from src.config import INITIAL_LEARNING_RATE, LEARNING_RATE_DECAY, RATIO_NB_NEURON, DIV_VARIANCE
from src.distance import select_closest
from src.io_helper import normalize
from src.neuron import generate_network, get_neighborhood, get_route
from src.plot import plot_network, plot_route
from src.utils import logger


def som(problem: 'pandas.DataFrame', iterations: int, plot=False) -> int:
    """
    Solve the TSP using a Self-Organizing Map.
    """
    learning_rate = INITIAL_LEARNING_RATE
    # Obtain the normalized set of cities (w/ coord in [0,1])
    cities = problem.copy()

    cities[['x', 'y']] = normalize(cities[['x', 'y']])

    # The population size is 8 times the number of cities
    n, p = cities.shape
    population_size = cities.shape[0] * RATIO_NB_NEURON

    # Generate an adequate network of neurons:
    network = generate_network(population_size)
    logger.info('Network of {} neurons created. Starting the iterations:'.format(n))

    cities_pts = cities[['x', 'y']].values
    for i in range(iterations):
        if not i % 500:
            logger.debug('> Iteration {}/{}'.format(i, iterations))
        # Choose a random city
        city = cities_pts[np.random.randint(n)]
        winner_idx = select_closest(network, city)
        # Generate a filter that applies changes to the winner's gaussian
        gaussian = get_neighborhood(winner_idx, n // DIV_VARIANCE, population_size)
        # Update the network's weights (closer to the city)
        network += gaussian[:, np.newaxis] * learning_rate * (city - network)
        # Decay the variables
        learning_rate = learning_rate * LEARNING_RATE_DECAY
        n = n * LEARNING_RATE_DECAY

        # Check for plotting interval
        if not i % 1000 and plot:
            plot_network(cities, network, name='diagrams/{:05d}.png'.format(i))

        # Check if any parameter has completely decayed.
        if n < 1:
            logger.warning('Radius has completely decayed, finishing execution at {} iterations'.format(i))
            break
        if learning_rate < 0.001:
            logger.warning('Learning rate has completely decayed, finishing execution at {} iterations'.format(i))
            break
    else:
        logger.info('Completed {} iterations.'.format(iterations))

    logger.debug('Getting routes')
    route = get_route(cities, network)

    if plot:
        plot_network(cities, network, name='diagrams/final.png')
        plot_route(cities, route, 'diagrams/route.png')
    return route
