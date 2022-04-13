# FARM
**F**oreground **A**ll-scale **R**adio **M**odeller.
## Requirements
- python 3.8+ (required for most up-to-date numpy library)
### Python modules
- [ARatmospy](https://github.com/SimonP2207/ARatmospy) (no version information available)
- [astropy](https://docs.astropy.org/en/stable/) (developed with v5.0)
- [h5py](https://docs.h5py.org/en/stable/) (developed with v3.6.0)
- [jupyter](https://jupyter.org/) (developed with v1.0.0)
- [matplotlib](https://matplotlib.org/) (developed with v3.5.0)
- [numpy](https://numpy.org/) (developed with v1.22.1)
- [oskarpy](https://fdulwich.github.io/oskarpy-doc) (no version information available)
- [pandas](https://pandas.pydata.org/) (developed with v1.4.1)
- [powerbox](https://powerbox.readthedocs.io/en/latest/) (developed with v0.6.1)
- [py21cmfast](https://21cmfast.readthedocs.io/en/latest/reference/py21cmfast.html) (no version information available)
- [pygdsm](https://github.com/telegraphic/pygdsm) (no version information available)
- [reproject](https://reproject.readthedocs.io/en/stable/) (developed with v0.8)
- [scipy](https://scipy.org/) (developed with v1.7.3)
- [toml](https://github.com/uiri/toml) (developed with v0.10.2)

### System
- [casa](https://casa.nrao.edu/) (developed with v6.4.0.16)
- [gsl](https://anaconda.org/conda-forge/gsl) (developed with v2.7)
- [miriad](https://www.atnf.csiro.au/computing/software/miriad/) (developed with v20220120)
- [oskar](https://ska-telescope.gitlab.io/sim/oskar/python/quickstart.html) (developed with v2.8.2)

## Installation help

For the python modules detailed above (and their dependencies (e.g. `gsl`), installation within a virtual environment is a possible method. Instructions for virtual environment creation and installing those dependencies listed above is shown below (requires a conda insallation).

**NOTE**: Installation of `py21cmFAST` on MacOSX leads to issues during compilation with the error, `clang: error: unsupported option '-fopenmp'` being thrown when installing via `pip install `. To properly resolve this and install `py21cmFAST` is detailed [here](https://github.com/21cmfast/21cmFAST/issues/84).

```commandline
conda create -n farm python>3.8
conda activate farm

# General package requirements
conda install astropy gsl h5py jupyter matplotlib numpy pandas scipy toml reproject

# Installation of reproject can fail, in which case:
conda install -c conda-forge reproject

# py21cmFAST
# OpenMP enabled libraries of fftw are available through conda-forge, using "conda install fftw" does not include openmp threads
conda install -c conda-forge fftw  # Only needed for MacOS
conda install -c conda-forge/label/cf201901 gcc  # Only needed for MacOS
conda install -c conda-forge 21cmFAST

# powerbox
conda install -c conda-forge pyfftw
conda install pip git
pip install git+git://github.com/steven-murray/powerbox.git

# Global diffuse sky model
pip install git+https://github.com/telegraphic/pygdsm

# OSKAR
pip install 'git+https://github.com/OxfordSKA/OSKAR.git@master#egg=oskarpy&subdirectory=python'

# Our fork of the ARatmospy repository for simulating the TEC screen
pip install git+https://github.com/SimonP2207/ARatmospy
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
from pathlib import Path
import astropy.units as u
from astropy.coordinates import SkyCoord
import farm

dims  = 512, 512  # n_pixels in x/R.A., n_pixels in y/declination
field_of_view = 8.  # deg
cell_size = field_of_view / dims[0]  # deg

coord = SkyCoord("01:02:03.4", "05:06:07.89",
                 frame='fk5', unit=(u.hourangle, u.degree))

# Use MHD model for Galactic small-scale structure SkyComponent. This is loaded 
# from data/Gsynch_SKAs.fits directly and thus its frequency information is 
# fixed.
gssm = farm.SkyComponent.load_from_fits(fitsfile=farm.DATA_FILES['MHD'],
                                        name='GSSM', cdelt=cell_size,
                                        coord0=coord)

# Use GSM2016 model (Zheng et al, 2016) for Galactic diffuse-scale structure 
# SkyComponent. Add the same frequencies as the GSSM sky component to enable 
# combination.
gdsm = farm.SkyComponent(dims, cell_size, coord,
                         tb_func=farm.tb_functions.gdsm2016_t_b)
gdsm.add_frequency(gssm.frequencies)

# Create SkyModel instance which will handle the combination of the various 
# SkyComponent instances. Add frequency information to it
skymodel = farm.SkyModel(dims, cell_size, coord, gssm.frequencies)

# Add individual SkyComponent instances to the SkyModel
skymodel += (gssm, gdsm)  # Equivalent to skymodel.add_component((gssm, gdsm))

# Write the sky model to a .fits file
skymodel.write_fits(Path("test_skymodel_Inu.fits"), unit='JY/SR')
```

## For co-developers
### Testing
For the unit-testing, Python's standard library package, `unittest` is used. 
Command line use is:
```commandline
python -m unittest /path/to/farm/tests/test_*.py
```
Otherwise, good IDEs (such as PyCharm) have inbuilt testing which can be set up 
for `unittest`. This is the preferred approach due to the inhuilt debugger which 
can be run in conjunction with the testing package.