from typing import TypeVar
from ._classes import SkyModel, SkyComponent, _BaseSkyClass

SkyClassType = TypeVar('SkyClassType', bound='_BaseSkyClass')
