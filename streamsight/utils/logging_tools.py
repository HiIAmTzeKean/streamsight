import logging
from typing import Literal
import warnings

def log_level(level: int):
    """
    Change the logging level for root logger.

    :param level: The new logging level (e.g., logging.DEBUG, logging.WARNING).
    :type level: int
    """
    logger = logging.getLogger()  # Root logger

    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
        
LOG_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}

def log_level_by_name(level_name: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"]):
    """
    Change the logging level using level names like 'DEBUG', 'INFO', etc.

    :param level_name: The name of the logging level (e.g., 'DEBUG').
    :type level_name: str
    """
    level = LOG_LEVELS.get(level_name.upper())
    if level is None:
        raise ValueError(f"Invalid logging level name: {level_name}")
    log_level(level)

def suppress_warnings(suppress: bool = True):
    """
    Enable or disable the suppression of warnings.

    :param suppress: If True, suppress all warnings. If False, allow warnings.
    :type suppress: bool
    """
    if suppress:
        logging.captureWarnings(False)  # Stops capturing warnings into logs
        warnings.filterwarnings("ignore")  # Suppresses all warnings
    else:
        logging.captureWarnings(True)  # Resumes capturing warnings into logs
        warnings.resetwarnings()  # Resets the filter to default behavior

def suppress_specific_warnings(category: type[Warning]):
    """
    Suppress specific warnings by category.

    :param category: The warning category to suppress (e.g., DeprecationWarning).
    :type category: type
    """
    warnings.filterwarnings("ignore", category=category)