import logging

from ata_logging import Logger

logging.getLogger("numba").setLevel(logging.INFO)

log = Logger(name="rolling_ta.pytest")
