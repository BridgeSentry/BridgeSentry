import datetime
from enum import Enum
import importlib
import sys

from torch_geometric import logging

from config.constants import Bridge

def log_to_file(message: str, log_file: str):
    """
    Writes an error message to the specified log file with a timestamp.

    :param message: The error message to be logged.
    :param log_file: Path to the log file (default: './error_log.log').
    """
    try:
        # Ensure the logger is configured once
        logger = logging.getLogger(log_file)
        if not logger.handlers:
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s\n")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.ERROR)

        # Log the error message
        logger.error(message)

    except Exception as e:
        # Fallback in case the logger fails
        with open(log_file, "a") as fallback_log:
            fallback_log.write(f"{datetime.now()} - ERROR - Failed to log error: {e}\n")
            fallback_log.write(f"{datetime.now()} - ERROR - Original message: {message}\n")


def log_error(bridge: Bridge, message: str):
    """
    Logs an error message to the console and writes it to a log file.

    :param message: The error message to be logged.
    """
    log_file = "./error_log.log"

    log_to_file(message, log_file)
    log_to_cli(build_log_message_generator(bridge, f"Error written to {log_file}"), CliColor.ERROR)

def build_log_message_generator(bridge: Bridge, message: str = ""):
    message = f"{datetime.now()} - INFO - {bridge.value} - {message}"

    return message


class CliColor(Enum):
    INFO = "\033[93m"
    SUCCESS = "\033[92m"
    WARNING = "\033[38;5;208m"
    ERROR = "\033[91m"


def log_to_cli(message: str = "", color: CliColor = CliColor.INFO):
    sys.stdout.write(f"{color.value}{message}\033[0m\n")


def get_enum_instance(enum_class: Enum, value: str):
    try:
        return enum_class(value.lower())
    except ValueError as e:
        raise ValueError(f"{value} is not a valid member of the {enum_class.__name__} Enum.") from e

def load_module(module_name: str):
    return importlib.import_module(module_name)

class CustomException(Exception):
    def __init__(self, classname: str, func_name: str, message: str):
        super().__init__(f"(Class: {classname}) {func_name}: {message}")