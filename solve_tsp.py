import click

from src.distance import route_distance
from src.io_helper import read_tsp
from src.main import som
from src.utils import logger


@click.command()
@click.argument('tsp-filepath')
@click.argument('iterations', type=int)
@click.option('--plot/--no-plot', default=False)
def main(tsp_filepath: str, iterations: int, plot: bool):
    logger.info('Reading TSP file: {}'.format(tsp_filepath))
    problem = read_tsp(tsp_filepath)

    logger.info('Starting searching sub-optimal solution')
    route = som(problem, iterations, plot=plot)

    logger.info('Reindexing DataFrame')
    problem = problem.reindex(route)

    logger.info('Fetch best path')
    distance = route_distance(problem)

    logger.info('Route found of length {}'.format(distance))


if __name__ == '__main__':
    main()
