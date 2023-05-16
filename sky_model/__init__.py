from typing import TypeVar
from ._classes import SkyModel, SkyComponent, SubbandSkyModel, _BaseSkyClass

SkyClassType = TypeVar('SkyClassType', bound='_BaseSkyClass')
