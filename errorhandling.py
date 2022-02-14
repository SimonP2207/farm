import warnings
from farm import LOGGER


def log_errors_warnings(function):
    """
    Decorator for logging and raising errors/warnings issued during method calls
    """
    def wrapper(*args, **kwargs):
        try:
            with warnings.catch_warnings(record=True) as w:
                result = function(*args, **kwargs)
                for wrng in [_.message for _ in w]:
                    issue_warning(wrng.category, wrng.args[0])
        except Exception as e:
            raise_error(e.__class__, e.args[0])
        return result
    return wrapper


def raise_error(error_type, error_message: str, log: bool = True):
    """Logs and raises an error"""
    if not issubclass(error_type, Exception):
        err_msg = "'error_type' must be an Wrror/Exception, not a " \
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
