import sys
import logging

logger = logging.getLogger(__name__)
loglevel = logging.INFO
logger.setLevel(loglevel)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

logger.propagate = False

if not any(isinstance(handler, logging.StreamHandler) and handler.stream == sys.stderr for handler in logger.handlers):
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(file_formatter)
    logger.addHandler(console_handler)

if not any(isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout for handler in logger.handlers):
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(loglevel)
    console_handler.addFilter(lambda record: record.levelno <= logging.INFO)
    console_handler.setFormatter(file_formatter)
    logger.addHandler(console_handler)
    