"""
Any and all decorator definitions should be placed in this module
"""
import warnings

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
