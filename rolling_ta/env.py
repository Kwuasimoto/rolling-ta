import os
from dotenv import load_dotenv

load_dotenv()


NUMBA_DISK_CACHING = True if os.getenv("NUMBA_DISK_CACHING") == 1 else False
