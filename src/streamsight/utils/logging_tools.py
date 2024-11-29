import logging
import logging.config
import os
import warnings
from collections.abc import Generator
from contextlib import contextmanager
from enum import Enum

import pyfiglet
import yaml

from streamsight.utils.path import safe_dir
from streamsight.utils.yaml_tool import create_config_yaml


class LogLevel(Enum):
    CRITICAL = logging.CRITICAL
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    INFO = logging.INFO
    DEBUG = logging.DEBUG
    NOTSET = logging.NOTSET

    @classmethod
    def from_string(cls, level: str) -> "LogLevel":
        """Return a LogLevel member from a case-insensitive string.

        Args:
            level: Name of the log level (case-insensitive).

        Returns:
            The corresponding :class:`LogLevel` enum member.
        """
        return cls[level.upper()]


def log_level(level: int | str | LogLevel) -> None:
    """Change the logging level for the root logger.

    Args:
        level: The logging level to set. May be a :class:`LogLevel` enum,
            a level name (str, case-insensitive), or an integer logging level.

    Returns:
        None
    """
    if isinstance(level, str):
        level = LogLevel.from_string(level)

    numeric_level = level.value if isinstance(level, LogLevel) else level

    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    for handler in logger.handlers:
        handler.setLevel(numeric_level)


log_level_by_name = log_level  # Alias for convenience


def suppress_warnings() -> None:
    """Suppress all Python warnings.

    This will disable warning output by filtering all warnings and
    disabling logging's capture of warnings.

    Returns:
        None
    """
    logging.captureWarnings(False)
    warnings.filterwarnings("ignore")


def enable_warnings() -> None:
    """Enable Python warnings (reset to default behavior).

    This re-enables warning capture and resets any filters previously set.

    Returns:
        None
    """
    logging.captureWarnings(True)
    warnings.resetwarnings()


def suppress_specific_warnings(category: type[Warning]) -> None:
    """Suppress warnings of a specific category.

    Args:
        category: Warning class/type to suppress (for example, :class:`DeprecationWarning`).

    Returns:
        None
    """
    warnings.filterwarnings("ignore", category=category)


@contextmanager
def warnings_suppressed() -> Generator:
    """Context manager that temporarily suppresses all warnings.

    Yields:
        None: Warnings are suppressed inside the context block.
    """
    logging.captureWarnings(False)
    warnings.filterwarnings("ignore")
    try:
        yield
    finally:
        logging.captureWarnings(True)
        warnings.resetwarnings()


def prepare_logger(log_config_filename: str) -> dict:
    """Prepare and configure logging from a YAML file.

    This function locates or creates a logging configuration YAML file using
    :func:`streamsight.utils.yaml_tool.create_config_yaml`, ensures the
    directory for the configured log file exists, writes an ASCII art header
    to the log file, and configures the Python logging system using
    :func:`logging.config.dictConfig`.

    Args:
        log_config_filename: Name of the logging configuration YAML file.

    Returns:
        dict: The parsed logging configuration dictionary.

    Raises:
        FileNotFoundError: If the resolved YAML configuration file cannot be found.
        ValueError: If there is an error parsing the YAML content.
    """
    _, yaml_file_path = create_config_yaml(log_config_filename)
    try:
        with open(yaml_file_path, "r") as stream:
            config = yaml.load(
                stream, Loader=yaml.FullLoader
            )
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {yaml_file_path}.")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")

    # Get the log file path from the configuration
    log_file = config["handlers"]["file"]["filename"]

    # Ensure the log file directory exists
    dir_name = os.path.dirname(log_file)
    safe_dir(dir_name)

    # Write ASCII art to the log file
    with open(log_file, "w") as log:
        ascii_art = pyfiglet.figlet_format("streamsight")
        log.write(ascii_art)
        log.write("\n")

    logging.config.dictConfig(config)
    logging.captureWarnings(True)
    return config
