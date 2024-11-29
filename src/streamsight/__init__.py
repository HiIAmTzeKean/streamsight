"""
Streamsight
-----------

Streamsight is a Python package toolkit developed for evaluation of recommendation
systems in different settings. Mainly the toolkit is developed to evaluate
in a sliding window setting.
"""

import logging

from streamsight.utils import prepare_logger


LOGGING_CONFIG_FILENAME = "logging_config.yaml"

prepare_logger(LOGGING_CONFIG_FILENAME)

logger = logging.getLogger(__name__)
logger.info("streamsight package loaded.")
