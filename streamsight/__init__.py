"""
Streamsight
-----------

Streamsight is a Python package toolkit developed for evaluation of recommendation
systems in different settings. Mainly the toolkit is developed to evaluate
in a sliding window setting.
"""

import logging
import logging.config

from streamsight.utils import prepare_logger

LOGGING_CONFIG = "LOGGING_CONFIG.yaml"

prepare_logger(LOGGING_CONFIG)

logger = logging.getLogger(__name__)
