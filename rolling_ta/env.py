import os
from dotenv import load_dotenv

from .logging import logger

load_dotenv()

NUMBA_DISK_CACHING = True if os.getenv("NUMBA_DISK_CACHING") == "1" else False
