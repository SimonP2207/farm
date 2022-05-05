"""
Any and all decorator definitions should be placed in this module
"""
import time
import logging
import warnings
from typing import Callable, Any

from .error_handling import issue_warning, raise_error


def docstring_parameter(*sub):
    def dec(obj):
        obj.__doc__ = obj.__doc__.format(*sub)
        return obj
    return dec


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


def suppress_warnings(*modules) -> Callable:
    """
    Suppress warnings issued by specified module(s)

    Parameters
    ----------
    modules
        Name(s) of modules as args

    Returns
    -------

    """
    def decorator(function: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            # In case the module warnings were ignored by prior, calling
            # function, as not to reset the ignored level
            ignored = [False] * len(modules)
            for idx, module in enumerate(modules):
                if logging.getLogger(module).level == logging.CRITICAL:
                    ignored[idx] = True

            # Raise Logger level and employ warning-filter
            for idx, module in enumerate(modules):
                if not ignored[idx]:
                    logging.getLogger(module).setLevel(logging.CRITICAL)
                    warnings.filterwarnings("ignore", module=module)

            # Execute function with args and kwargs
            result = function(*args, **kwargs)

            # Reset Logger level and remove warning-filter
            for idx, module in enumerate(modules):
                if not ignored[idx]:
                    logging.getLogger(module).setLevel(logging.WARNING)
                    warnings.filterwarnings("default", module=module)

            return result
        return wrapper
    return decorator
