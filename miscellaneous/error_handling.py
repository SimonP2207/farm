"""
Functions designed to allow farm to raise exceptions, issue warnings and log
either event, expecially in the error case whereby logs are written before the
exception terminates the program
"""
import warnings
from .. import LOGGER


def raise_error(error_type, error_message: str, log: bool = True):
    """Logs and raises an error"""
    if not issubclass(error_type, Exception):
        err_msg = "'error_type' must be an Error/Exception, not a " \
                  f"{type(error_type)}. Attempted error message was " \
                  f"{error_message}"
        if log:
            LOGGER.error(err_msg)
        raise TypeError(err_msg)
    if log:
        LOGGER.error(f"{error_type.__name__}:: {error_message}")
    raise error_type(error_message)


def issue_warning(warning_type, warning_message: str, log: bool = True):
    """Logs and issues an error"""
    if not issubclass(warning_type, Warning):
        err_msg = "'warning_type' must be a Warning type, not a " \
                  f"{type(warning_type)}. Attempted warning message was " \
                  f"{warning_message}"
        if log:
            LOGGER.warning(err_msg)
        raise TypeError(err_msg)
    if log:
        LOGGER.warning(f"{warning_type.__name__}:: {warning_message}")
    warnings.warn(warning_message, warning_type)