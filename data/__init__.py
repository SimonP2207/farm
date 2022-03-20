import numpy as np
from pathlib import Path
from . import loader

DATA_FILE_DCY = Path(__file__).parent.joinpath('files')
DATA_FILES = {'ATEAM': 'ateam.data',
              'EXAMPLE_CONFIG': 'example_config.toml',
              'MHD': 'Gsynch_SKAs.fits', }
DATA_FILES = {k: DATA_FILE_DCY / v for k, v in DATA_FILES.items()}

for identifier, data_file in DATA_FILES.items():
    if not data_file.exists():
        raise FileNotFoundError(f"{identifier} data file, {str(data_file)}, "
                                f"does not exist")

# excluded_files = ('CONTENTS.md',)
# for file in DATA_FILE_DCY.glob('**/*'):
#     if file not in DATA_FILES.values() and file.name not in excluded_files:
#         warnings.warn(f"{str(file)} not listed in farm.data.DATA_FILES",
#                       Warning)
# del excluded_files

ATEAM_DATA = np.loadtxt(DATA_FILES['ATEAM'], delimiter=',')
