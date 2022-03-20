"""
Foreground All-scale Radio Modeller (FARM)
"""
import pathlib
import logging; LOGGER = logging.getLogger(__name__)
from . import sky_model

__version__ = (0, 0, 1)
__all__ = ['LOGGER']

# Define directories within the farm package
_farm_loc = pathlib.Path(__file__).parent
