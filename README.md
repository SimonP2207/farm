# FARM
**F**oreground **A**ll-scale **R**adio **M**odeller.
## Requirements
- [numpy]()
- [reproject]()
- [scipy]()
- [jupyter]()
- [matplotlib]()
- [astropy]()
- [h5py]()
- [gsl]()
- [py21cmfast]()
- [powerbox]()
- [oskar](https://ska-telescope.gitlab.io/sim/oskar/python/quickstart.html)

## Installation

Installation within a virtual environment is the safest and most fool-proof method.

Installation of `py21cmFAST` on MacOSX leads to issues during compilation with the error, `clang: error: unsupported option '-fopenmp'` being thrown when installing via `pip install `. To properly resolve this and install `py21cmFAST` is detailed [here](https://github.com/21cmfast/21cmFAST/issues/84).

```bash
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

