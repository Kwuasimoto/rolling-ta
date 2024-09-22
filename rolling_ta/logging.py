import logging

numba_logger = logging.getLogger("numba")
logger = logging.getLogger("rolling-ta")

logging.basicConfig(
    level=logging.INFO,
    # format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    format="%(levelname)s - %(message)s",
)
