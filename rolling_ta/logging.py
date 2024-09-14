import logging

numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.DEBUG,
    # format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    format="%(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
