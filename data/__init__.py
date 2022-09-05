"""Contains all methods for loading and handling data"""
from pathlib import Path
from typing import Callable
import pandas as pd

from . import loader

__all__ = ['loader', 'FILES', 'DATA_FILE_DCY', 'ATEAM_DATA']


def _iterate_through_dict(d: dict, operation: Callable) -> dict:
    for k, v in d.items():
        if isinstance(v, dict):
            _iterate_through_dict(v, operation)
        else:
            d[k] = operation(v)
    return d


def _check_file_exists(f: Path):
    if not f.exists():
        raise FileNotFoundError(f"{f} does not exist")
    return f

DATA_DCY = Path(__file__).parent
DATA_FILE_DCY = Path(__file__).parent / 'files'

FILES = {
    'EXAMPLE_CONFIG': 'example_config.toml',
    'IMAGES': {
        'MHD': 'Gsynch_SKAs.fits',
        'TRECS': 'sky_continuum_sdc3_v1_1.fits',
    },
    'TABLES': {
        'ATEAM': 'ateam.data',
        'GLEAM': 'GLEAM_catalogue.fits',
        'LOBES': 'GLM_LoBESv2.osm',
        'EOR_PARAMS': 'eor_param_data.json'
    },
}

FILES = _iterate_through_dict(FILES, lambda v: DATA_FILE_DCY / v)
FILES = _iterate_through_dict(FILES, _check_file_exists)

FILES['IMAGES']['MHD2'] = Path('/my/path/to/mhd2.fits')

# excluded_files = ('CONTENTS.md',)
# for file in DATA_FILE_DCY.glob('**/*'):
#     if file not in DATA_FILES.values() and file.name not in excluded_files:
#         warnings.warn(f"{str(file)} not listed in farm.data.DATA_FILES",
#                       Warning)
# del excluded_files
# ATEAM_DATA = np.loadtxt(DATA_FILES['ATEAM'], delimiter=',')

ATEAM_DATA = pd.read_csv(FILES['TABLES']['ATEAM'], sep=",", comment='#',
                         skipinitialspace=True)
