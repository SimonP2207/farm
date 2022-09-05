from pathlib import Path
from ..data import FILES, DATA_DCY

__all__ = ['DATA_FILES', 'IMAGE_FILES']

DATA_FILES = {'eor_params': FILES['TABLES']['EOR_PARAMS']}
IMAGE_FILES = {'skao_logo': DATA_DCY / 'images' / 'skao_logo.png'}
