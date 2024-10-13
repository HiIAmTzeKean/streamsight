"""
Streamsight
-----------

Streamsight is a Python package toolkit developed for evaluation of recommendation
systems in different settings. Mainly the toolkit is developed to evaluate
in a sliding window setting.
"""

import logging
import logging.config

from streamsight.utils import (log_level, log_level_by_name, prepare_logger,
                               suppress_specific_warnings, suppress_warnings)

LOGGING_CONFIG_FILENAME = "LOGGING_CONFIG.yaml"

prepare_logger(LOGGING_CONFIG_FILENAME)

logger = logging.getLogger(__name__)
logger.info("streamsight package loaded.")
