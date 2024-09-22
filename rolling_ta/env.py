import os
from dotenv import load_dotenv

load_dotenv()

NUMBA_DISK_CACHING = True if os.getenv("NUMBA_DISK_CACHING") == "1" else False
NUMBA_PARALLEL = True if os.getenv("NUMBA_PARALLEL") == "1" else False
NUMBA_FASTMATH = True if os.getenv("NUMBA_FASTMATH") == "1" else False
NUMBA_NOGIL = True if os.getenv("NUMBA_NOGIL") == "1" else False
