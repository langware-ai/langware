import logging

console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s (%(asctime)s) [%(name)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console_handler.setFormatter(formatter)

logger = logging.getLogger('promptchain')
logger.addHandler(console_handler)

# logger.setLevel(logging.DEBUG)  # Set logging level to DEBUG to capture all log messages
