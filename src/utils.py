import logging

logger = logging.getLogger('som-tsp')
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S-%f',
                    level=logging.DEBUG)
