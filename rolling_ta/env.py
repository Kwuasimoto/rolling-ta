from ata_config import cfg

NUMBA_DISK_CACHING = True if cfg.get("NUMBA_DISK_CACHING") == "1" else False
NUMBA_PARALLEL = True if cfg.get("NUMBA_PARALLEL") == "1" else False
NUMBA_FASTMATH = True if cfg.get("NUMBA_FASTMATH") == "1" else False
NUMBA_NOGIL = True if cfg.get("NUMBA_NOGIL") == "1" else False
