# FARM
**F**oreground **A**ll-scale **R**adio **M**odeller.
## Requirements
- python 3.6+ (f-string dependency)
### Python modules
- [astropy](https://docs.astropy.org/en/stable/) (developed with v5.0)
- [h5py](https://docs.h5py.org/en/stable/) (developed with v3.6.0)
- [jupyter](https://jupyter.org/) (developed with v1.0.0)
- [matplotlib](https://matplotlib.org/) (developed with v3.5.0)
- [numpy](https://numpy.org/) (developed with v1.22.1)
- [oskarpy](https://fdulwich.github.io/oskarpy-doc) (no version information available)
- [powerbox](https://powerbox.readthedocs.io/en/latest/) (developed with v0.6.1)
- [py21cmfast](https://21cmfast.readthedocs.io/en/latest/reference/py21cmfast.html) (no version information available)
- [pygdsm](https://github.com/telegraphic/pygdsm) (no version information available)
- [reproject](https://reproject.readthedocs.io/en/stable/) (developed with v0.8)
- [scipy](https://scipy.org/) (developed with v1.7.3)
- [toml](https://github.com/uiri/toml) (developed with v0.10.2)

### System
- [casa](https://casa.nrao.edu/) (developed with v6.4.0.16)
- [gsl](https://anaconda.org/conda-forge/gsl) (developed with v2.7)
- [miriad](https://www.atnf.csiro.au/computing/software/miriad/)(developed with v20220120)
- [oskar](https://ska-telescope.gitlab.io/sim/oskar/python/quickstart.html) (developed with v2.8.2)

## Installation help

Installation within a virtual environment is an easy method, for which instructions on installing those dependencies listed above is shown below.

Installation of `py21cmFAST` on MacOSX leads to issues during compilation with the error, `clang: error: unsupported option '-fopenmp'` being thrown when installing via `pip install `. To properly resolve this and install `py21cmFAST` is detailed [here](https://github.com/21cmfast/21cmFAST/issues/84).

```commandline
conda create -n farm python=3
conda activate farm

# General package requirements
conda install numpy scipy jupyter matplotlib astropy h5py gsl

# py21cmFAST
# OpenMP enabled libraries of fftw are available through conda-forge, using "conda install fftw" does not include openmp threads
conda install -c conda-forge fftw
conda install -c conda-forge/label/cf201901 gcc
conda install -c conda-forge 21cmFAST

# powerbox
conda install -c conda-forge pyfftw
conda install pip git
pip install git+git://github.com/steven-murray/powerbox.git

# Global diffuse sky model
pip install git+https://github.com/telegraphic/pygdsm

# OSKAR
pip install 'git+https://github.com/OxfordSKA/OSKAR.git@master#egg=oskarpy&subdirectory=python'
```

### A note on miriad and casa
Both miriad and casa limit the possible number of antennae in an array.

## Basic Use
### As a CLI application
```commandline
python3 /path/to/farm/main.py /path/to/config.toml
```
### As an imported library
```python
import farm
import astropy.units as u
from astropy.coordinates import SkyCoord

dims  = 512, 512  # n_pixels in x/R.A., n_pixels in y/declination
field_of_view = 8.  # deg
cell_size = field_of_view / dims[0]  # deg

coord = SkyCoord("01:02:03.4", "05:06:07.89",
                 frame='fk5', unit=(u.hourangle, u.degree))

# Use MHD model for Galactic small-scale structure SkyComponent. This is loaded 
# from data/Gsynch_SKAs.fits directly and thus its frequency information is 
# fixed.
gssm = farm.GSSM(dims, cell_size, coord, model='MHD')

# Use GSM2016 model (Zheng et al, 2016) for Galactic diffuse-scale structure 
# SkyComponent. Add the same frequencies as the GSSM sky component to enable 
# combination.
gdsm = farm.GDSM(dims, cell_size, coord, model='GSM2016')
gdsm.add_frequency(gssm.frequencies)

# Create SkyModel instance which will handle the combination of the various 
# SkyComponent instances. Add frequency information to it
skymodel = farm.SkyModel(dims, cell_size, coord)
skymodel.add_frequency(gssm.frequencies)

# Add individual SkyComponent instances to the SkyModel
skymodel.add_component(gssm)
skymodel.add_component(gdsm)

```
