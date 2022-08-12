"""
Any and all decorator definitions should be placed in this module
"""
import inspect
from functools import wraps
import logging
import warnings
from typing import Callable, Any, Tuple


def ensure_is_fits(*files):
    """Ensure the listed arguments are fits files"""
    def decorator(func):
        """decorator function"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            """wrapper function"""
            from . import fits
            argspec = inspect.getfullargspec(func)
            for f in files:
                if argspec.kwonlyargs and f in argspec.kwonlyargs:
                    val = kwargs[f]
                else:
                    val = args[argspec.args.index(f)]
                if not fits.is_fits(val):
                    raise TypeError(f"{val} not a .fits file")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def docstring_parameter(*sub):
    """Insert values into object documentation"""
    def wrapper(obj):
        """wrapper function"""
        obj.__doc__ = obj.__doc__.format(*sub)
        return obj
    return wrapper


def log_errors_warnings(function):
    """
    Decorator for logging and raising errors/warnings issued during method calls
    """
    from .error_handling import issue_warning, raise_error

    def wrapper(*args, **kwargs):
        """wrapper function"""
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
        """decorator function"""
        def wrapper(*args, **kwargs) -> Any:
            """wrapper function"""
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


def time_it(func: Callable) -> Callable:
    """
    Time the execution of a function and return a 2-tuple of time [s] taken and
    returned values
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Tuple[float, Any]:
        """Decorated function"""
        from datetime import datetime

        t_start = datetime.now()
        returned = func(*args, **kwargs)
        t_end = datetime.now()
        dt = (t_end - t_start).total_seconds()

        return dt, returned
    return wrapper
