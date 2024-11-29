import os

import yaml

from .path import get_logs_dir, safe_dir


def create_config_yaml(config_filename: str) -> tuple[str, str]:
    """
    Create a configuration file for the logger.

    Writes a default configuration file for the logger in YAML format.
    The configuration file specifies the format of the log messages,
    the output stream, and the log level.

    :param path: The name of the file to be created.
    :type path: str
    """
    log_dir = get_logs_dir()
    safe_dir(log_dir)
    log_file_path = os.path.join(log_dir, "streamsight.log")
    yaml_file_path = os.path.join(log_dir, config_filename)

    # check if log file and yaml file already exist
    if os.path.exists(yaml_file_path) and os.path.exists(log_file_path):
        return (log_file_path, yaml_file_path)

    default_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
            "simple": {"format": "%(levelname)s - %(message)s"},
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "DEBUG",
                "formatter": "simple",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.FileHandler",
                "level": "DEBUG",
                "formatter": "detailed",
                "filename": log_file_path,
            },
        },
        "loggers": {
            "streamsight": {
                "level": "DEBUG",
                "handlers": ["console", "file"],
                "propagate": False,
            },
        },
        "root": {
            "level": "WARNING",
            "handlers": ["console"],
        },
    }

    # Write the YAML content
    with open(yaml_file_path, "w") as file:
        yaml.dump(default_config, file, default_flow_style=False)
    return (log_file_path, yaml_file_path)
