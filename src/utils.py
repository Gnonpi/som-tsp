import logging

logger = logging.getLogger('som-tsp')
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)
