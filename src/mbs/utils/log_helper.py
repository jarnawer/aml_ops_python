import logging


def get_logger(name):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    return logger
