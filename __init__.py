import pathlib
import logging

__version__ = (0, 0, 1)
__all__ = ['LOGGER', 'DATA_FILES']

# Define directories within the farm package
_pkg_loc = pathlib.Path(__file__).parent
_data_dcy = _pkg_loc.joinpath('data')

# Set up the package-level logger
LOGGER = logging.getLogger(__name__)
DATA_FILES = {'MHD': _data_dcy.joinpath('Gsynch_SKAs.fits'),
              'ATEAM': _data_dcy.joinpath('ateam.data')}

from .classes import GDSM, GSSM, SkyModel
from . import plotting_functions
