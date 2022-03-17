"""
farm.miscellaneous subpackage contains methods not logically belonging to the
other farm subpackages, or not suited to placement in farm's main directory
"""
from . import decorators
from . import error_handling


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
