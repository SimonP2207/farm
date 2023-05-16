"""
Any and all decorator definitions should be placed in this module
"""
import inspect
import logging
import warnings
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterable, Tuple


def convert_str_to_path(args_to_convert: Iterable[str]) -> Callable:
    """
    Converts argument(s) from str to Path. arg_to_convert can be a str or list
    of strings, each string being the name of an argument of the decorated
    function to convert from str to Path. In case the argument value is None, no
    conversion is performed
    """
    if isinstance(args_to_convert, str):
        args_to_convert = [args_to_convert]

    def decorator(func: Callable) -> Callable:
        """decorator function"""
        from functools import wraps
        from inspect import getfullargspec

        func_argspec = getfullargspec(func)
        func_args, func_kwonlyargs = func_argspec.args, func_argspec.kwonlyargs

        for arg_to_convert in args_to_convert:
            if arg_to_convert not in func_args + func_kwonlyargs:
                raise ValueError(f"{arg_to_convert} not in args or kwargs "
                                 f"of {func.__name__}")

        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            """wrapper function"""

            def _str_to_path(x: Any) -> Any:
                if isinstance(x, str):
                    return Path(x).resolve()
                elif isinstance(x, Iterable):
                    for i, v in enumerate(x):
                        x[i] = _str_to_path(v)
                return x

            for arg_to_convert in args_to_convert:
                if arg_to_convert in kwargs.keys():
                    val = kwargs[arg_to_convert]
                    # if isinstance(val, str):
                    kwargs[arg_to_convert] = _str_to_path(val)
                    # kwargs[arg_to_convert] = Path(val).resolve()

                elif arg_to_convert in func_args:
                    idx_arg = func_args.index(arg_to_convert)
                    try:
                        val = args[idx_arg]
                    except IndexError:  # Gets thrown in optional arg case
                        continue
                    # if isinstance(val, str):
                    args = list(args)
                    args[idx_arg] = _str_to_path(val)
                    # args[idx_arg] = Path(val).resolve()
                    args = tuple(args)

            return func(*args, **kwargs)

        return wrapper

    return decorator


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

    @wraps(function)
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

        @wraps(function)
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
