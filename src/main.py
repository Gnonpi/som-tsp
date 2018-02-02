import numpy as np

from src.distance import select_closest
from src.io_helper import normalize
from src.neuron import generate_network, get_neighborhood, get_route
from src.plot import plot_network, plot_route
from src.utils import logger


def som(problem: 'pandas.DataFrame', iterations: int, learning_rate: float = 0.8, plot=False) -> int:
    """
    Solve the TSP using a Self-Organizing Map.
    """

    # Obtain the normalized set of cities (w/ coord in [0,1])
    cities = problem.copy()

    cities[['x', 'y']] = normalize(cities[['x', 'y']])

    # The population size is 8 times the number of cities
    n = cities.shape[0] * 8

    # Generate an adequate network of neurons:
    network = generate_network(n)
    logger.info('Network of {} neurons created. Starting the iterations:'.format(n))

    for i in range(iterations):
        if not i % 100:
            logger.debug('> Iteration {}/{}'.format(i, iterations))
        # Choose a random city
        city = cities.sample(1)[['x', 'y']].values
        winner_idx = select_closest(network, city)
        # Generate a filter that applies changes to the winner's gaussian
        gaussian = get_neighborhood(winner_idx, n // 10, network.shape[0])
        # Update the network's weights (closer to the city)
        network += gaussian[:, np.newaxis] * learning_rate * (city - network)
        # Decay the variables
        learning_rate = learning_rate * 0.99997
        n = n * 0.9997

        # Check for plotting interval
        if not i % 1000 and plot:
            plot_network(cities, network, name='diagrams/{:05d}.png'.format(i))

        # Check if any parameter has completely decayed.
        if n < 1:
            logger.warning('Radius has completely decayed, finishing execution',
                           'at {} iterations'.format(i))
            break
        if learning_rate < 0.001:
            logger.warning('Learning rate has completely decayed, finishing execution',
                           'at {} iterations'.format(i))
            break
    else:
        logger.info('Completed {} iterations.'.format(iterations))

    logger.debug('Getting routes')
    route = get_route(cities, network)

    if plot:
        plot_network(cities, network, name='diagrams/final.png')
        plot_route(cities, route, 'diagrams/route.png')
    return route
