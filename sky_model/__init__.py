from typing import TypeVar
from .classes import SkyModel, SkyComponent, _BaseSkyClass

SkyClassType = TypeVar('SkyClassType', bound='_BaseSkyClass')
