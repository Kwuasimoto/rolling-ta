import os


NUMBA_DISK_CACHING = True if os.getenv("NUMBA_DISK_CACHING") == 1 else False
