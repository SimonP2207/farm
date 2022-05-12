from pathlib import Path
from typing import Callable
import pandas as pd

__all__ = ['loader']

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


def fits_table_to_dataframe(fits_table: Path) -> pd.DataFrame:
    from astropy.io import fits

    with fits.open(fits_table) as hdugsm:
        data = pd.DataFrame.from_records(hdugsm[1].data)

    return data


DATA_FILE_DCY = Path(__file__).parent.joinpath('files')
FILES = {
    'EXAMPLE_CONFIG': 'example_config.toml',
    'IMAGES': {
        'MHD': 'Gsynch_SKAs.fits',
        'TRECS': 'sky_continuum_sdc3_v1_1.fits',
    },
    'TABLES': {
        'ATEAM': 'ateam.data',
        'GLEAM': 'GLEAM_EGC_v2_compact.fits'
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
