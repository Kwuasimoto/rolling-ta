from logging import Logger

import pytest


def test_obv(log):
    if isinstance(log, Logger):
        log.info("Logger loaded!")
