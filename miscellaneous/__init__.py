"""
farm.miscellaneous subpackage contains methods not logically belonging to the
other farm subpackages, or not suited to placement in farm's main directory
"""
from typing import Union
import numpy as np
import numpy.typing as npt


def generate_random_chars(length: int, choices: str = 'alphanumeric') -> str:
    """
    For generating sequence of random characters for e.g. file naming

    Parameters
    ----------
    length
        Number of characters to generate
    choices
        Characters to choose from. Can be 'alphanumeric', 'alpha', or 'numeric.
        Default is 'alphanumeric
    Returns
    -------
    string of defined length comprised of random characters from desired
    character range

    Raises
    ------
    ValueError
        If 'choices' not one of 'alphanumeric', 'alpha' or 'numeric'
    """
    import random

    if choices not in ('alphanumeric', 'alpha', 'numeric'):
        raise ValueError("choices must be one of 'alphanumeric', 'alpha', or "
                         f"'numeric', not {choices}")
    poss_chars = ''
    if 'alpha' in choices:
        poss_chars += ''.join([chr(_) for _ in range(65, 91)])
        poss_chars += ''.join([chr(_) for _ in range(97, 123)])
    if 'numeric' in choices:
        poss_chars += ''.join([chr(_) for _ in range(49, 58)])

    assert poss_chars, "Not sure how poss_chars is an empty string..."

    return ''.join([random.choice(poss_chars) for _ in range(length)])


def interpolate_values(desired_x: float,
                       y_lo: Union[float, npt.ArrayLike],
                       y_hi: Union[float, npt.ArrayLike],
                       x_lo: float,
                       x_hi: float,
                       nans_ok: bool = False) -> npt.ArrayLike:
    """
    Calculate the interpolated value(s) between two values/array of
    values to a desired x-value, assuming a constant power-law. In cases
    where negative y-values are present, take the mean of the two
    (non-log) y-values if desired, else return nan values

    Parameters
    ----------
    desired_x
        x-value to interpolate to
    y_lo
        y-value(s) at the low x-value
    y_hi
        y-alue(s) at the high x-value
    x_lo
        Low x-value
    x_hi
        High x-value
    nans_ok
        Whether it is ok to have nans in the resultant value/array of
        values. False by default, in which case the geometric mean
        replaces the nan

    Returns
    -------
    Interpolated y-value(s) at desired x-value
    """
    # Calculate logarithm value(s) of all args
    log_y_lo, log_y_hi = np.log10(y_lo), np.log10(y_hi)
    log_x_lo, log_x_hi = np.log10([x_lo, x_hi])
    log_desired_x = np.log10(desired_x)

    # Calculate power-law coefficient(s)
    coeff = (log_y_hi - log_y_lo) / (log_x_hi - log_x_lo)

    log_y = coeff * (log_desired_x - log_x_lo) + log_y_lo
    desired_y = np.power(10., log_y)

    # In cases where negative fluxes were originally present, a nan is
    # generated. In those cases just take the mean of the corresponding
    # values instead, if requested
    if not nans_ok:
        desired_y = np.where(
            np.isnan(desired_y), (y_hi - y_lo) / 2 + y_lo, desired_y
        )

    return desired_y
