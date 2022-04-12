from .miriad import miriad
from .common import which
from . import oskar

# Check installations of casa, miriad and oskar
for program in ('miriad', 'casa', 'oskar'):
    if which(program) is None:
        raise SystemError(f"{program} not in PATH")
