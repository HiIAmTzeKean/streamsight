import os


def safe_dir(path):
    """Check if directory is safe
    
    Check if the directory exists, if not create it.

    :param path: The path to the directory.
    :type path: str
    """
    if not os.path.exists(path):
        os.makedirs(path)


def create_config_yaml(filename):
    """
    Create a configuration file for the logger.
    
    Writes a default configuration file for the logger in YAML format.
    The configuration file specifies the format of the log messages,
    the output stream, and the log level.
    
    :param filename: The name of the file to be created.
    :type filename: str
    """
    content = """\
version: 1
formatters:
  colored:
    (): colorlog.ColoredFormatter
    format: '%(log_color)s%(levelname)-8s%(reset)s - %(name)s - %(blue)s%(message)s'
  simple:
    format: '%(asctime)s: - %(name)s - %(levelname)s - %(message)s'
    datefmt: "%Y-%m-%d %H:%M:%S"
    class: logging.Formatter
handlers:
  console:
    class: logging.StreamHandler
    level: DEBUG
    formatter: colored
    stream: ext://sys.stdout
  file:
    class: logging.FileHandler
    level: DEBUG
    formatter: simple
    filename: ./log/output.log
loggers:
  root:
    level: DEBUG
    handlers: [console,file]
"""
    with open(filename, "w") as f:
        f.write(content)
