"""
Foreground All-scale Radio Modeller (FARM)
"""
import pathlib
import warnings
import logging; LOGGER = logging.getLogger(__name__)
from . import (calibration, data, miscellaneous, observing, physics, sky_model,
               software)

LOG_FMT = "%(asctime)s:: %(levelname)s:: %(module)s.%(funcName)s:: %(message)s"
LOG_DATE_FMT = "%Y-%m-%d %H:%M:%S"

# Define directories within the farm package
_farm_loc = pathlib.Path(__file__).parent

__version__ = (0, 0, 1)
__all__ = ['LOGGER', 'calibration', 'data', 'miscellaneous', 'observing',
           'physics', 'sky_model', 'software']


# def _append_to_pathlib_path(self, suffix):
#     return self.parent / (self.name + suffix)
#
#
# pathlib.Path.append = _append_to_pathlib_path

pathlib.Path.append = lambda self, suffix: self.parent / (self.name + suffix)


# Disable warnings from erfa module
logging.getLogger('erfa').setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", module='erfa')
