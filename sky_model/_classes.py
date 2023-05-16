"""
All classes for use within the FARM infrastructure.
"""
import copy
import shutil
import pathlib
import tempfile
import math
from abc import ABC, abstractmethod
from typing import Iterable, Tuple, List, Optional, Union, TypeVar, Type

import numpy.typing as npt
import numpy as np
import pandas as pd
import astropy.units as u
import scipy.signal
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io.fits import Header

from ..miscellaneous import error_handling as errh
from ..miscellaneous import decorators
from . import tb_functions as tbfs

# Typing related code
SkyClassType = TypeVar('SkyClassType', bound='_BaseSkyClass')


# Miscellaneous functions
# TODO: Relocate these to a sensible module
def deconvolve_cube(input_fits, output_fits, beam):
    """Deconvolves a .fits image with a beam. DO NOT USE: Unstable results"""
    from ..software.miriad import miriad
    from ..miscellaneous import generate_random_chars

    temp_identifier = generate_random_chars(10)
    input_mir_im = pathlib.Path(input_fits.name.rstrip('.fits'))
    temp_mir_im = pathlib.Path(f"{input_mir_im}_temp_{temp_identifier}")

    miriad.fits(_in=f"'{input_fits}'",
                out=f"'{input_mir_im}'",
                op='xyin')

    miriad.convol(map=f"'{input_mir_im}'",
                  out=f"'{temp_mir_im}'",
                  options="divide",
                  fwhm=f"{beam[0] * 3600.},{beam[1] * 3600.}",
                  pa=f"{beam[2]}",
                  sigma=1)
    shutil.rmtree(input_mir_im)

    miriad.fits(_in=f"'{temp_mir_im}'",
                out=f"'{output_fits}'",
                op='xyout')
    shutil.rmtree(temp_mir_im)


def hdr2d_from_skymodel(sky_class: Type[SkyClassType]) -> Header:
    """SubbandSkyModel or SkyComponent 2D (RA and declination) header"""
    from ..miscellaneous.fits import hdr2d

    return hdr2d(sky_class.n_x, sky_class.n_y, sky_class.coord0,
                 sky_class.cdelt)


def hdr3d_from_skyclass(sky_class: Type[SkyClassType]) -> Header:
    """SubbandSkyModel or SkyComponent 3D (RA, declination and frequency) header"""
    from ..miscellaneous.fits import hdr3d

    if len(sky_class.frequencies) < 1:
        raise ValueError("Can't create Header from SkyClass with no frequency "
                         "information")

    return hdr3d(sky_class.n_x, sky_class.n_y, sky_class.coord0,
                 sky_class.cdelt, sky_class.frequencies)


def deconvolve_fwhm(conv_size: float, beam_size: float) -> float:
    """Deconvolved FWHM dimensions given a beam size and convolved FWHM size"""
    return np.sqrt(conv_size ** 2. - beam_size ** 2.)


class _BaseSkyClass(ABC):
    """
    Abstract superclass for all subclasses representing sky models (i.e.
    distributions of temperature brightness/intensity/flux on the celestial
    sphere)

    Class Attributes
    ----------------
    _FREQ_TOL : float
        Tolerance (in Hz) determining if two frequencies are the same. See
        frequency_present method
    VALID_UNITS : tuple of str
        Units that are valid outputs of t_b/i_nu/flux_nu methods and valid
        inputs from .fits files

    Attributes
    ----------
    n_x : int
        Number of pixels in the x (right ascension) direction
    n_y : int
        Number of pixels in the x (right ascension) direction
    cdelt: float
        Pixel size [deg]
    coord0: astropy.coordinates.SkyCoord
        Field-of-view's central coordinate
    """
    from ..observing import Subband

    _FREQ_TOL = 1.
    VALID_UNITS = ('K', 'JY/PIXEL', 'JY/SR', 'JY/BEAM')

    @classmethod
    def load_from_oskar_sky_model(cls, osmfile: pathlib.Path,
                                  name: str, cdelt: float,
                                  coord0: SkyCoord, fov: Tuple[float, float],
                                  freqs: Union[Subband, npt.ArrayLike],
                                  flux_range: Tuple[float, float] = (0., 1e30),
                                  default_spix: float = -0.7,
                                  beam: Optional[dict] = None
                                  ) -> 'SkyComponent':
        """
        Create a SkyComponent instance from an Oskar sky model
        (https://ska-telescope.gitlab.io/sim/oskar/sky_model/sky_model.html#sky-model-file)

        Parameters
        ----------
        osmfile
            Full path to oskar sky model text file
        name
            Name to give SkyComponent
        cdelt
            Cell size [deg]
        coord0
            Central pixel coordinate
        fov
            Field of view as a tuple (fov_x, fov_y) [deg]
        freqs
            Frequencies of the SkyComponent [Hz]
        flux_range
            Flux range to INCLUDE [Jy]
        beam
            Beam with which to deconvolve dimensions as a dict i.e.
            {'maj': 120, 'min': 120., 'pa': 0.} [deg]

        Returns
        -------
        SkyComponent instance
        """
        from ..miscellaneous.file_handling import osm_to_dataframe

        if hasattr(freqs, 'frequencies'):
            freqs = freqs.frequencies

        return cls.load_from_dataframe(osm_to_dataframe(osmfile), name, cdelt,
                                       coord0, fov, freqs, flux_range,
                                       default_spix, beam)

    @classmethod
    def load_from_dataframe(cls, df: pd.DataFrame,
                            name: str, cdelt: float,
                            coord0: SkyCoord, fov: Tuple[float, float],
                            freqs: Union[Subband, npt.ArrayLike],
                            flux_range: Tuple[float, float] = (0., 1e30),
                            default_spix: float = -0.7,
                            beam: Optional[dict] = None) -> 'SkyComponent':
        """
        Create a SkyComponent instance from pandas.DataFrame instance

        Parameters
        ----------
        df
            pandas.DataFrame with columns 'ra', 'dec', 'fluxI', 'fluxQ',
            'fluxU', 'fluxV', 'freq0', 'spix', 'RM', 'maj', 'min', and 'pa'
        name
            Name to give SkyComponent
        cdelt
            Cell size [deg]
        coord0
            Central pixel coordinate
        fov
            Field of view as a tuple (fov_x, fov_y) [deg]
        freqs
            Frequencies of the SkyComponent [Hz]
        flux_range
            Flux range within to include sources [Jy]
        default_spix
            Default spectral index to assign to sources without one listed.
            Default is -0.7 typical of extragalactic sources
        beam
            Beam with which to deconvolve dimensions as a dict i.e.
            {'maj': 0.012, 'min': 0.012., 'pa': 0.} [deg]

        Returns
        -------
        SkyComponent instance
        """
        from astropy import wcs
        from ..physics import astronomy as ast
        from ..miscellaneous import image_functions as imfunc
        from ..miscellaneous import generate_random_chars
        from ..miscellaneous.file_handling import dataframe_to_osm
        from ..miscellaneous.fits import hdr3d

        if hasattr(freqs, 'frequencies'):
            freqs = freqs.frequencies

        # #################################################################### #
        # ################# Clean and manipulate DataFrame ################### #
        # #################################################################### #
        # Remove sources outside field of view or accepted flux range
        flux_range_mask = ((df['fluxI'] >= flux_range[0]) &
                           (df['fluxI'] <= flux_range[1]))

        fov_mask = ast.within_square_fov(
            fov, coord0.ra.deg, coord0.dec.deg, df.ra, df.dec
        )

        df = df.drop(df.index[~(fov_mask & flux_range_mask)])

        # Assign default spectral index to those sources lacking one
        df.loc[np.isnan(df.spix), 'spix'] = default_spix

        # Deconvolve given sizes from beam, if given
        if beam is not None:
            if not np.isclose(beam['maj'], beam['min'], rtol=1e-2):
                errh.raise_error(ValueError,
                                 "Circular beams only for deconvolution of "
                                 "point source table ")

            df['maj'] = deconvolve_fwhm(df['maj'], beam['maj'])
            df['min'] = deconvolve_fwhm(df['min'], beam['min'])

            # Default size for sources smaller than the beam
            for col in ('maj', 'min', 'pa'):
                df.loc[np.isnan(df[col]), col] = 0.

        # #################################################################### #
        # ################# Create image header and data grid ################ #
        # #################################################################### #
        # Set up fits header, WCS and data array
        im_hdr = hdr3d(int(fov[0] // cdelt),
                       int(fov[1] // cdelt),
                       coord0, cdelt, freqs)
        im_hdr.insert('CUNIT3', ('BUNIT', 'JY/PIXEL'), after=True)
        im_wcs = wcs.WCS(im_hdr)
        naxis1, naxis2, naxis3 = im_wcs.pixel_shape
        im_data = np.zeros((naxis3, naxis2, naxis1))

        # Set up pixel grid (xx, yy, zz) and coordinate grid (rra, ddec, ffreq)
        rra, ddec, ffreq = imfunc.make_coordinate_grid(im_wcs)

        # Individually add each source to grid of fluxes (Jy/pixel) by
        for _, row in df.iterrows():
            # Width in indices of sub-array within data to calculate source flux
            if (row['maj'] < cdelt * 3600) and (row['min'] < cdelt * 3600):
                imfunc.place_point_source_on_grid(
                    im_data, im_wcs, row.ra, row.dec, row.fluxI, row.freq0,
                    row.spix
                )
            else:
                imfunc.place_gaussian_on_grid(
                    im_data, rra, ddec, ffreq, im_wcs,
                    row.ra, row.dec, row.fluxI, row.freq0,
                    row.spix, row['maj'], row['min'], row['pa']
                )

        # Generate temporary .fits image to call load_from_fits method on
        hdu = fits.PrimaryHDU(data=im_data, header=im_hdr)
        temp_identifier = generate_random_chars(10)
        temp_fits_file = pathlib.Path(f"{name}_temp_{temp_identifier}.fits")
        hdu.writeto(temp_fits_file, overwrite=True)

        sky_comp = cls.load_from_fits(temp_fits_file, name, cdelt, coord0)
        temp_fits_file.unlink()

        sky_comp.oskar_table = pathlib.Path(f'{name}.osm')
        dataframe_to_osm(df, sky_comp.oskar_table)

        return sky_comp

    @classmethod
    def load_from_fits_table(cls, fitsfile: Union[pathlib.Path, fits.HDUList],
                             name: str, cdelt: float,
                             coord0: SkyCoord, fov: Tuple[float, float],
                             freqs: Union[Subband, npt.ArrayLike],
                             flux_range: Tuple[float, float] = (0., 1e30),
                             default_spix: float = -0.7,
                             beam: Optional[dict] = None) -> 'SkyComponent':
        """
        Creates a SkyComponent instance from a .fits table file or HDUList
        loaded from such

        Parameters
        ----------
        fitsfile
            pathlib.Path to .fits table, or HDUList instance
        name
            Name to give returned SkyComponent instance
        cdelt
            Cell size [deg]
        coord0
            Central coordinate
        fov
            Field of view extent in x and y as a tuple [deg]
        freqs
            Frequencies of the SkyComponent
        flux_range
            Lower and upper bound of source fluxes as a 2-tuple, (lower, upper)
        default_spix
            Default spectral index to assign to sources without one listed.
            Default is -0.7 typical of extragalactic sources
        beam
            Beam with which catalogue sizes are convolved with (dict). If
            specified, the beam will be deconvolved from the source dimensions
            before placing within the brightness distribution. Ensure that beam
            major and minor axes are in the same units as the fits table's
            source major/minor axes. Default is None (i.e. no deconvolution).
            Circular beams only are allowed for the moment. Example of required
            dict format:
                {'maj': 0.01, 'min': 0.01, 'pa': 0.}

        Returns
        -------
        SkyComponent instance

        Raises
        ------
        TypeError
            If fitsfile is not a pathlib.Path or HDUList instance

        ValueError
            If non-circular beam is specified as the beam argument
        """
        from ..miscellaneous.fits import fits_table_to_dataframe

        if hasattr(freqs, 'frequencies'):
            freqs = freqs.frequencies

        if isinstance(fitsfile, pathlib.Path):
            data = fits_table_to_dataframe(fitsfile)
        elif isinstance(fitsfile, fits.HDUList):
            data = pd.DataFrame.from_records(fitsfile[1].data)
        else:
            data = None
            errh.raise_error(TypeError, f"{fitsfile} not an HDUList instance "
                                        "or path to a .fits table")

        sky_comp = cls.load_from_dataframe(
            data, name, cdelt, coord0, fov, freqs, flux_range, default_spix,
            beam
        )

        return sky_comp

    @classmethod
    @decorators.docstring_parameter(str(VALID_UNITS)[1:-1])
    def load_from_fits(
            cls, fitsfile: pathlib.Path,
            name: Optional[str] = None,
            cdelt: Optional[float] = None,
            coord0: Optional[SkyCoord] = None,
            freqs: Optional[Union[npt.NDArray, Subband]] = None
    ) -> 'SkyComponent':
        """
        Creates a SkyComponent instance from a .fits cube

        Parameters
        ----------
        fitsfile
            Full path to .fits file
        name
            Name to assign to the output instance. If not given, the fits
            filename (without the file type) will be given by default
        cdelt
            Pixel size [deg]. By default this is parsed from the .fits header.
            Note that specifying a value artifically changes pixel size i.e.
            fluxes (i.e. Jy/pixel) stay the same from the input .fits to the
            output t_b data
        coord0
            Central coordinate (corresponds to the fits-header CRVAL1 and
            CRVAL2 keywords) as a astropy.coordinates.SkyCoord instance. By
            default this is parsed from the .fits header
        freqs
            Desired frequencies for SkyComponent. If not identical to that found
            in the .fits header, values will be interpolated. However, all
            desired frequencies must be within the frequency coverage of the
            .fits

        Returns
        -------
        SkyComponent instance

        Raises
        ------
        FileNotFoundError
            If fitsfile doesn't exist

        ValueError
            If the units found in the .fits header are not one of {0} or are not
            understood
        """
        from .. import LOGGER
        from ..physics import astronomy as ast
        from ..miscellaneous.fits import (fits_hdr_and_data, fits_frequencies,
                                          fits_equinox, fits_bunit)

        if hasattr(freqs, 'frequencies'):
            freqs = freqs.frequencies

        LOGGER.info(f"Loading {str(fitsfile.resolve())}")
        if not fitsfile.exists():
            errh.raise_error(FileNotFoundError,
                             f"{str(fitsfile.resolve())} not found")

        fits_hdr, fits_data = fits_hdr_and_data(fitsfile)

        if fits_data.ndim != 3:
            errh.raise_error(ValueError, ".fits must be a cube (3D), but has "
                                         f"{fits_data.ndim} dimensions")

        # Image information
        fits_freqs = fits_frequencies(fitsfile)
        equinox = fits_equinox(fitsfile)  # 1950.0 or 2000.0
        unit = fits_bunit(fitsfile)  # Brightness unit information
        nx, ny = fits_hdr["NAXIS1"], fits_hdr["NAXIS2"]

        if not unit:
            raise ValueError("No brightness unit information present in "
                             f"{str(fitsfile)}")

        if unit not in cls.VALID_UNITS:
            raise ValueError("Please ensure unit is one of "
                             f"{repr(cls.VALID_UNITS)[1:-1]}, not "
                             f"'{unit}'")

        if cdelt is None:
            cdelt = fits_hdr["CDELT2"]

        if coord0 is None:
            frame = {1950.0: 'fk4', 2000.0: 'fk5'}[equinox]
            coord0 = SkyCoord(fits_hdr["CRVAL1"], fits_hdr["CRVAL2"],
                              unit=(u.deg, u.deg), frame=frame)

        if freqs is None:
            freqs = fits_freqs  # Spectral axis information
            data = fits_data
        else:
            from ..miscellaneous import interpolate_values

            freqs.sort()  # Ensure freqs are sorted numerically
            data = np.empty((len(freqs), ny, nx), dtype=np.float32)
            for idx, freq in enumerate(freqs):
                if any(np.isclose(fits_freqs, freq, atol=cls._FREQ_TOL)):
                    fits_idx = np.argwhere(np.isclose(
                        fits_freqs, freq, atol=cls._FREQ_TOL
                    ))
                    fits_idx = fits_idx.flatten()[0]
                    data[idx] = fits_data[fits_idx]
                else:
                    # Find at which index the freq would have to be inserted
                    # into fits_freq to maintain sorted numerical order
                    fits_data_idx = np.searchsorted(fits_freqs, freq)

                    # Throw error if desired frequencies lie outside of the
                    # .fits' frequency coverage
                    if fits_data_idx in (0, len(fits_freqs)):
                        raise ValueError(
                            "Desired frequencies go past .fits frequency "
                            "coverage. Only interpolation (not extrapolation) "
                            "of fluxes is supported"
                        )
                    else:
                        data[idx] = interpolate_values(
                            freq,
                            fits_data[fits_data_idx - 1],
                            fits_data[fits_data_idx],
                            fits_freqs[fits_data_idx - 1],
                            fits_freqs[fits_data_idx]
                        )

        if unit == 'K':
            pass
        elif unit == 'JY/SR':
            data = ast.intensity_to_tb(data, freqs)
        elif unit == 'JY/PIXEL':
            data = ast.flux_to_tb(data, freqs, np.radians(cdelt) ** 2.)

        if not name:
            name = fitsfile.name.strip('.fits')[0]

        sky_model = SkyComponent(name, (nx, ny), cdelt=cdelt, coord0=coord0,
                                 tb_func=tbfs.fits_t_b)
        sky_model._frequencies = freqs
        sky_model._tb_data = data

        return sky_model

    def __init__(self, n_pix: Tuple[int, int], cdelt: float, coord0: SkyCoord):
        """
        Parameters
        ----------
        n_pix
            Number of pixels in x (R.A.) and y (declination) as a 2-tuple
        cdelt
            Pixel size [deg]
        coord0
            Central coordinate (corresponds to the fits-header CRVAL1 and
            CRVAL2 keywords) as a astropy.coordnates.SkyCoord instance
        """
        # Attributes created from constructor args
        self.n_x, self.n_y = n_pix
        self.cdelt = cdelt
        self.coord0 = coord0

        # Private instance attributes
        self._frequencies = np.array([])
        self._tb_data = np.empty((len(self._frequencies), self.n_y, self.n_x),
                                 dtype=np.float32)
        # Call __post_init__ equivalent to dataclass' __post_init__
        self.__post_init__()

        self.oskar_table: Optional[pathlib.Path] = None

    def __post_init__(self):
        pass

    @decorators.docstring_parameter(str(VALID_UNITS)[1:-1])
    def data(self, unit: str) -> npt.ArrayLike:
        """
        Sky distribution image cube

        Parameters
        ----------
        unit
            One of {0}
        Returns
        -------
            np.ndarray of dimensions (len(self.frequencies), self.n_y, self.n_x)
            containing sky brightness distribution in units of 'unit'

        Raises
        ------
        ValueError
            If the given unit is not one of {0}, or is not understood
        """
        from ..physics import astronomy as ast

        data_ = None
        if unit not in self.VALID_UNITS:
            errh.raise_error(ValueError,
                             f"Something has gone unexpectedly wrong. "
                             f"Unit in loaded .fits image is '{unit}'")
        elif unit == 'K':
            data_ = self._tb_data
        elif unit == 'JY/PIXEL':
            data_ = ast.tb_to_flux(self._tb_data, self.frequencies,
                                   np.radians(self.cdelt) ** 2.)
        elif unit == 'JY/SR':
            data_ = ast.tb_to_intensity(self._tb_data, self.frequencies)
        else:
            raise ValueError(f"{unit} not valid as unit")

        return data_

    @property
    def frequencies(self):
        """Observational frequencies"""
        return self._frequencies

    @frequencies.setter
    def frequencies(self, new_freq):
        self.add_frequency(new_freq)

    def add_frequency(self, new_freq: Union[float, npt.ArrayLike, Subband]):
        """
        Add an observing frequency (or frequencies)

        Parameters
        ----------
        new_freq
            Observing frequency to add. Can be a float, iterable of floats, or
            Subband instance from which frequncies will be retrieved

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If one of frequencies/the frequency being added is not a float
        """
        if hasattr(new_freq, 'frequencies'):
            new_freq = new_freq.frequencies

        if isinstance(new_freq, (list, tuple, np.ndarray)):
            for freq in new_freq:
                self.add_frequency(freq)
        elif isinstance(new_freq, (float, np.floating)):
            # Check frequency not already present
            if np.isclose(self._frequencies, new_freq).any():
                wrng_msg = f"Supplied frequency, {new_freq:.3f}Hz, already " \
                           f"present"
                errh.issue_warning(UserWarning, wrng_msg)
            else:
                self._frequencies = np.append(self._frequencies, new_freq)
                self._frequencies.sort()
                idx = np.asarray(self._frequencies == new_freq).nonzero()[0][0]
                self._tb_data = np.insert(self._tb_data, idx,
                                          self.t_b(new_freq), axis=0)
        else:
            err_msg = f"Can't add a {type(new_freq)} to frequencies," \
                      f" must be a float or list-like object containing floats"
            errh.raise_error(TypeError, err_msg)

    @property
    def header(self):
        """Sky class' 3D (RA, declination and frequency) header"""
        return hdr3d_from_skyclass(self)

    @property
    def header2d(self):
        """Sky class' 2D (RA and declination) header"""
        return hdr2d_from_skymodel(self)

    @abstractmethod
    def t_b(self, freq: Union[float, npt.ArrayLike]) -> npt.NDArray:
        """
        Calculate brightness temperature distribution [K]

        Parameters
        ----------
        freq
           Frequency at which brightness temperatures are wanted [Hz]

        Returns
        -------
        np.ndarray of brightness temperatures with shape (self.n_x, self.n_y)
        """

    def i_nu(self, freq: Union[float, npt.ArrayLike]) -> npt.NDArray:
        """
        Calculate intensity sky distribution [Jy/sr]

        Parameters
        ----------
        freq
           Frequency at which brightness temperatures are wanted [Hz]

        Returns
        -------
        np.ndarray of intensities with shape (self.n_x, self.n_y)
        """
        from ..physics import astronomy as ast

        return ast.tb_to_intensity(self.t_b(freq), freq)

    def flux_nu(self, freq: Union[float, npt.ArrayLike]) -> npt.NDArray:
        """
        Calculate fluxes sky distribution [Jy/pixel]

        Parameters
        ----------
        freq
           Frequency at which fluxes are wanted [Hz]

        Returns
        -------
        np.ndarray of fluxes with shape (self.n_x, self.n_y)
        """
        from ..physics import astronomy as ast

        solid_angle = np.radians(self.cdelt) ** 2.

        return ast.intensity_to_flux(self.i_nu(freq), solid_angle)

    @decorators.docstring_parameter(str(VALID_UNITS)[1:-1])
    def write_fits(self, fits_file: pathlib.Path, unit: str,
                   beam: Optional[Tuple[float, float]] = None,
                   convolve: bool = False):
        """
        Write .fits cube of intensities or brightness temperatures

        Parameters
        ----------
        fits_file
            Full path to write .fits file to
        unit
            One of {0}
        beam
            Beam major FWHM [arcsec], minor FWHM [arcsec] and position angle
            [deg] as a 3-tuple. Only needed if unit is JY/BEAM
        convolve
            Whether to convolve the data by the given beam. Default is False

        Returns
        -------
        None

        Raises
        ------
        ValueError
            When supplied unit is not one of {0}
        """
        from .. import LOGGER
        # TODO: Implement convolution by the beam, if requested. Use
        #  scipy.signal.convolve2d
        LOGGER.info(f"Generating .fits file, {str(fits_file)}")

        if unit not in self.VALID_UNITS:
            err_msg = f"{unit} not a valid unit. Choose one of " \
                      f"{str(self.VALID_UNITS)[1:-1]}"
            errh.raise_error(ValueError, err_msg)

        hdr = self.header
        hdr.set('BUNIT', format(unit, '8'))

        if unit == 'JY/BEAM':
            if beam is None or len(beam) != 3:
                errh.raise_error(
                    ValueError,
                    f"Beam must be 3-tuple of (BMAJ, BMIN, BPA) , not '{beam}'"
                )
            gsmhdu = fits.PrimaryHDU(self.data(unit='JY/SR'))
            solid_angle_beam = beam[0] * beam[1] * 2.663263603293828e-11  # sr
            gsmhdu.data *= solid_angle_beam
            hdr.set('BMAJ', beam[0] / 3600.)
            hdr.set('BMIN', beam[1] / 3600.)
            hdr.set('BPA', beam[2])

            if convolve:
                # TODO: Move the below code to a sensible module
                def fwhm_to_sd(fwhm):
                    """FWHM to standard deviation for a Gaussian distribution"""
                    return fwhm / (2. * np.sqrt(2. * np.log(2.)))

                def gaussian2d(x, y, x0, y0, amp, sx, sy, pa):
                    """
                    2D Gaussian distribution

                    Parameters
                    ----------
                    x
                        x-coordinate to calculate function at
                    y
                        y-coordinate to calculate function at
                    x0
                        x-coordinate of Gaussian peak
                    y0
                        y-coordinate of Gaussian peak
                    amp
                        Peak value of Gaussian
                    sx
                        Standard deviation in x
                    sy
                        Standard deviation in y
                    pa
                        Position angle of Gaussian major axis [deg]

                    Returns
                    -------
                    f(x,y)
                    """
                    cospasq, sinpasq = np.cos(pa) ** 2., np.sin(pa) ** 2.
                    sin2pa = np.sin(2. * pa)

                    a = cospasq / (2. * sx ** 2.) + sinpasq / (2. * sy ** 2.)
                    b = sin2pa / (4. * sx ** 2.) - sin2pa / (4. * sy ** 2.)
                    c = sinpasq / (2. * sx ** 2.) + cospasq / (2. * sy ** 2.)

                    return amp * np.exp(-(
                            a * (x - x0) ** 2 + 2. * b * (x - x0) * (
                            y - y0) + c * (y - y0) ** 2.))

                xy_max = beam[0] * 2 / 3600.

                sx, sy = fwhm_to_sd(beam[0] / 3600), fwhm_to_sd(beam[1] / 3600)
                nxy = np.ceil(xy_max / self.cdelt)

                xy_pix = np.arange(-nxy, nxy, 1)
                xy = xy_pix * self.cdelt
                response = gaussian2d(xy[:, np.newaxis], xy[np.newaxis, :],
                                      0., 0., 1., sx, sy, np.radians(beam[2]))
                response /= np.nansum(response)

                for idx in range(len(gsmhdu.data)):
                    gsmhdu.data[idx] = scipy.signal.convolve2d(
                        gsmhdu.data[idx], response, mode='same'
                    )

        else:
            gsmhdu = fits.PrimaryHDU(self.data(unit=unit))

        gsmhdu.header = hdr

        if fits_file.exists():
            wrng_msg = f"{fits_file} already exists. Overwriting"
            errh.issue_warning(UserWarning, wrng_msg)

        gsmhdu.writeto(fits_file, overwrite=True)

    @decorators.docstring_parameter(str(VALID_UNITS)[1:-1])
    def write_miriad_image(self, miriad_image: pathlib.Path, unit: str):
        """
        Write miriad image of intensities or brightness temperatures

        Parameters
        ----------
        miriad_image
            Full path to write miriad image to
        unit
            One of {0}

        Returns
        -------
        None

        Raises
        ------
        FileExistsError
            In the unlikely event that the randomly generated name for a
            created, temporary file already exists (very unlikely!)
        """
        from ..software import miriad
        from ..miscellaneous import generate_random_chars

        temp_dir = pathlib.Path(tempfile.gettempdir())
        temp_pfx = temp_dir.joinpath(str(miriad_image.name).replace('.', '_'))
        temp_identifier = generate_random_chars(10)
        temp_fits_file = pathlib.Path(f"{temp_pfx}_temp_{temp_identifier}.fits")

        if temp_fits_file.exists():
            raise FileExistsError("I don't know how this is possible. There "
                                  "was literally had a 1 in 800 quadrillion "
                                  "chance of randomly generating the same "
                                  "name as a file that already exists")

        self.write_fits(temp_fits_file, unit=unit)

        if miriad_image.exists():
            errh.issue_warning(UserWarning,
                               f"{str(miriad_image)} already exists, removing")
            shutil.rmtree(miriad_image)

        miriad.fits(_in=temp_fits_file, out=miriad_image, op='xyin')
        temp_fits_file.unlink()

    @abstractmethod
    def summary(self) -> str:
        """Return a handy summary as a str and also print it"""
        ...

    def regrid(self, template: Type[SkyClassType]) -> SkyClassType:
        """
        Regrid the SkyClass onto the coordinate grid of another SkyClass. Make
        sure that the SkyClass to be regridded is in the field of view of the
        template SkyClass, or a flat field of zero intensity will be found. Uses
        miriad's regrid task to perform the regridding.

        Parameters
        ----------
        template
            SkyClass instance which to grid ONTO

        Returns
        -------
        New sky model regridded onto template grid
        """
        from ..software import miriad
        from ..miscellaneous import generate_random_chars

        # TODO: Implement check here that self is in field of view of template

        # Define temporary images
        # Fail if frequency information doesn't match
        if not self.same_spectral_setup(template):
            errh.raise_error(ValueError,
                             "Frequency information of sky class instance to be"
                             " regridded does not match that of the template")

        sffx = generate_random_chars(10)
        template_mir_image = pathlib.Path(f'temp_template_image_{sffx}.im')
        input_mir_image = pathlib.Path(f'temp_input_image_{sffx}.im')
        out_mir_image = pathlib.Path(f'temp_out_image_{sffx}.im')
        out_mir_image_imblr = pathlib.Path(f'temp_out_image_imblr_{sffx}.im')
        out_fits_image = pathlib.Path(f'out_image_{sffx}.fits')

        temporary_images = (template_mir_image, input_mir_image,
                            out_mir_image, out_mir_image_imblr,
                            out_fits_image)

        for temporary_image in temporary_images:
            if temporary_image.exists():
                shutil.rmtree(temporary_image)

        template.write_miriad_image(template_mir_image, unit='K')
        self.write_miriad_image(input_mir_image, unit='K')

        miriad.regrid(_in=input_mir_image,
                      tin=template_mir_image,
                      out=out_mir_image)

        miriad.imblr(_in=out_mir_image, out=out_mir_image_imblr, value=0.)
        miriad.fits(op='xyout', _in=out_mir_image_imblr, out=out_fits_image)

        regridded_input_temp = self.load_from_fits(out_fits_image)

        for temporary_image in temporary_images:
            if temporary_image.is_dir():
                shutil.rmtree(temporary_image)
            else:
                temporary_image.unlink()

        regridded_input_skyclass = copy.deepcopy(self)
        regridded_input_skyclass.coord0 = regridded_input_temp.coord0
        regridded_input_skyclass.n_x = regridded_input_temp.n_x
        regridded_input_skyclass.n_y = regridded_input_temp.n_y
        regridded_input_skyclass.cdelt = regridded_input_temp.cdelt
        regridded_input_skyclass._tb_data = regridded_input_temp._tb_data

        return regridded_input_skyclass

    def rotate(self, angle: float,
               inplace: bool = True) -> Optional[SkyClassType]:
        """
        Rotate the sky brightness distribution by specified angle

        Parameters
        ----------
        angle
            Angle with which to rotate brightness distribution
            counter-clockwise [radians]
        inplace
            Whether to rotate the sky brightness distribution of the original
            instance (inplace=True) or return a new instance with the rotated
            brightness data (inplace=False). Default is True

        Returns
        -------
        None if inplace=True, or SkyComponent instance if inplace=False
        """
        from ..miscellaneous.image_functions import rotate_image
        rotated_data = rotate_image(self._tb_data, angle, x_axis=2, y_axis=1)

        if inplace:
            self._tb_data = rotated_data
        else:
            new_skymodel_type = copy.deepcopy(self)
            new_skymodel_type._tb_data = rotated_data
            return new_skymodel_type

    def frequency_present(self, freq: float,
                          abs_tol: float = _FREQ_TOL) -> Union[float, bool]:
        """
        Check if frequency already present in SubbandSkyModel

        Parameters
        ----------
        freq
            Frequency to check [Hz]
        abs_tol
            Absolute tolerance to determine two matching frequencies [Hz]
        Returns
        -------
        False if no matching frequency found, or the matching frequency itself
        """
        for freq_1 in self.frequencies:
            if math.isclose(freq, freq_1, abs_tol=abs_tol):
                return freq_1

        return False

    def possess_similar_header(self, other: Type[SkyClassType]) -> bool:
        """
        Check if matching image sizes, frequencies and cell sizes between this
        and another sky class instance. Acceptable difference in cell sizes is
        the difference between the two pixel sizes for which a misalignment of
        the pixel grids across the whole image of less than half a pixel would
        be seen

        Parameters
        ----------
        other
            Other sky class instance for comparison

        Returns
        -------
        True if similar header information, False otherwise
        """
        if self.n_x != other.n_x:
            errh.issue_warning(UserWarning,
                               "n_x not matching for self and other, i.e. "
                               f"{self.n_x} != {other.n_x}")
            return False
        if self.n_y != other.n_y:
            errh.issue_warning(UserWarning,
                               "n_y not matching for self and other, i.e. "
                               f"{self.n_y} != {other.n_y}")
            return False
        if not all([other.frequency_present(f) for f in self.frequencies]):
            errh.issue_warning(UserWarning,
                               "frequencies not matching for self and other, "
                               f"i.e. {self.frequencies} != "
                               f"{other.frequencies}")
            return False

        rel_pix_tol = 0.5 / max([self.n_x, self.n_y])
        if not math.isclose(self.cdelt, other.cdelt, rel_tol=rel_pix_tol):
            errh.issue_warning(UserWarning,
                               "cdelt not matching for self and other, i.e. "
                               f"{self.cdelt} != {other.cdelt}")
            return False

        if not math.isclose(self.coord0.ra.deg, other.coord0.ra.deg,
                            rel_tol=rel_pix_tol):
            errh.issue_warning(UserWarning,
                               "Central RA coordinate not matching for self and"
                               f"other, i.e.{self.coord0.ra.deg:.5f}deg != "
                               f"{other.coord0.ra.deg:.5f}deg")
            return False

        if not math.isclose(self.coord0.dec.deg, other.coord0.dec.deg,
                            rel_tol=rel_pix_tol):
            errh.issue_warning(UserWarning,
                               "Central Dec coordinate not matching for self "
                               f"and other, i.e."
                               f"{self.coord0.dec.deg:.5f}deg != "
                               f"{other.coord0.dec.deg:.5f}deg")
            return False

        return True

    def same_spectral_setup(self,
                            other: Union[Type[SkyClassType],
                                         'Correlator']) -> bool:
        """
        Check if matching frequencies are present in two sky class instances

        Parameters
        ----------
        other
            Other sky class instance for comparison

        Returns
        -------
        True if same frequency information, False otherwise
        """
        if len(self.frequencies) != len(other.frequencies):
            return False

        if not all(np.isclose(self.frequencies, other.frequencies, 1.)):
            return False

        return True


class SkyComponent(_BaseSkyClass):
    """
    Class holding a single sky component which describes a single abstraction
    on the sky e.g. point sources, diffuse Galactic emission etc.
    """

    def __init__(self, name: str, npix: Tuple[int, int], cdelt: float,
                 coord0: SkyCoord, tb_func: tbfs.TbFunction):
        """
        Parameters
        ----------
        name
            Name to assign to SkyComponent instance
        npix
            Number of pixels in x (R.A.) and y (declination) as a 2-tuple
        cdelt
            Pixel size [deg]
        coord0
            Central coordinate (corresponds to the fits-header CRVAL1 and
            CRVAL2 keywords) as a astropy.coordnates.SkyCoord instance
        tb_func
            Callable to describe brightness temperature distribution on the sky
            [K]. Must take only 2 arguments being sky_component (SkyComponent
            instance) and freq (observing frequency, float or array of floats)
            and return a numpy array of brightness temperatures
        """
        super().__init__(npix, cdelt, coord0)
        self.name = name
        self._tb_func = tb_func

    @property
    def format(self) -> str:
        """
        Either 'table' or 'image' indicating whether this is a tabular ('table')
        or image-based ('image') SkyComponent instance
        """
        return 'table' if self.oskar_table else 'image'

    def summary(self):
        """Return a handy summary as a str and also prints it"""
        smry = (f"{self.__class__.__name__} instance, '{self.name}', in "
                f"{self.format} format")

        if self.format == 'table':
            from ..miscellaneous.file_handling import osm_to_dataframe

            df = osm_to_dataframe(self.oskar_table)
            n_points = np.nansum((df.maj == 0).values |
                                 (df['min'] == 0).values)
            n_gaussians = np.nansum((df.maj != 0).values &
                                    (df['min'] != 0).values)

            smry += (f" consisting of ({n_gaussians} Gaussians and "
                     f"{n_points} point-like sources ({len(df)} total) "
                     f"written to {self.oskar_table.resolve()}")
        else:
            smry += (f", over {len(self.frequencies)} channels from "
                     f"{min(self.frequencies) / 1e6:.1f}-"
                     f"{max(self.frequencies) / 1e6:.1f}MHz on an image grid "
                     f"of {self.n_x}\u00D7{self.n_y}, {self.cdelt * 3600:.1f} "
                     f"arcsec pixels")

        print(smry)

        return smry

    def normalise(self, other: Type[SkyClassType],
                  inplace: bool = False) -> Optional[SkyClassType]:
        """
        Adjust the SkyComponent instance's brightness data in order to properly
        recover the angular power spectrum of the combined power spectrum of
        this and another SkyComponent instance

        Parameters
        ----------
        other
            Other SkyComponent instance to normalise to. This should be the
            SkyComponent defining the low end of the angular power spectrum
        inplace
            Whether to normalise the sky brightness distribution of the original
            instance (inplace=True) or return a new instance with the normalised
            brightness data (inplace=False). Default is False
        Returns
        -------
        None if inplace=True, or SkyComponent instance if inplace=False
        """
        self_sum_td_nu = np.nansum(self.data(unit='JY/PIXEL'), axis=(1, 2))
        other_sum_td_nu = np.nansum(other.data(unit='JY/PIXEL'), axis=(1, 2))

        scalings = (other_sum_td_nu / self_sum_td_nu)[:, np.newaxis, np.newaxis]

        if inplace:
            self._tb_data *= scalings

        else:
            new_skymodeltype = copy.deepcopy(self)
            new_skymodeltype._tb_data *= scalings
            return new_skymodeltype

    def merge(self, other: Type[SkyClassType],
              beam: Optional[Tuple[float, float, float]] = None,
              other_beam: Optional[Tuple[float, float, float]] = None,
              new_name: Optional[str] = None,
              normalise: bool = True) -> SkyClassType:
        """
        Merge this SkyComponent with another, lower-resolution SkyComponent,
        whilst preserving power on all scales. Total fluxes per channel in the
        resultant image are normalised to the total fluxes per channel of the
        low-resolution image if wanted

        Parameters
        ----------
        other
            Other SkyComponent instance to merge with. This should be the
            'low-resolution' component
        beam
            Beam/angular resolution of this (high-resolution) SkyComponent in
            degrees as a 3-tuple of floats -> (bmaj, bmin, bpa). This will
            replace any beam information. If None, ensure beam information is
            present
        other_beam
            Beam/angular resolution of the other (low-resolution) SkyComponent
            in degrees as a 3-tuple of floats -> (bmaj, bmin, bpa). This will
            replace any beam information. If None, ensure beam information is
            present
        new_name
            Name to assigned to new, merged SkyComponent instance. If None
            (default), name will be a combination of the two SkyComponent
            instance names
        normalise
            Adjust the total fluxes per channel of the resultant merged image to
            the total fluxes per channel of the low-resolution image if True
            (default)

        Returns
        -------
        New SkyComponent instance which is the flux/power-preserving merger of
        the two SkyComponent instances
        """
        from pathlib import Path
        from ..software import miriad
        from ..miscellaneous import generate_random_chars

        if new_name is None:
            new_name = f"{self.name}+{other.name}"

        # Define all image names
        temp_id = generate_random_chars(10)
        mir_im_self = Path(f"{self.name.replace(' ', '_')}_{temp_id}.im")
        mir_im_other = Path(f"{other.name.replace(' ', '_')}_{temp_id}.im")
        mir_im_merge = Path(f"{new_name.replace(' ', '_')}_{temp_id}.im")
        fits_merge = Path(mir_im_merge.name.replace('.im', '.fits'))

        self.write_miriad_image(mir_im_self, unit='JY/PIXEL')
        other.write_miriad_image(mir_im_other, unit='JY/PIXEL')

        # Put beam information in fits headers for immerge if specified in args
        if other_beam is not None:
            miriad.puthd(_in=f"{mir_im_other}/bmaj",
                         value=f"{other_beam[0]:.6f},deg")
            miriad.puthd(_in=f"{mir_im_other}/bmin",
                         value=f"{other_beam[1]:.6f},deg")
            miriad.puthd(_in=f"{mir_im_other}/bpa",
                         value=f"{other_beam[2]:.2f},deg")

        if beam is not None:
            miriad.puthd(_in=f"{mir_im_self}/bmaj",
                         value=f"{beam[0]:.6f},deg")
            miriad.puthd(_in=f"{mir_im_self}/bmin",
                         value=f"{beam[1]:.6f},deg")
            miriad.puthd(_in=f"{mir_im_self}/bpa",
                         value=f"{beam[2]:.2f},deg")

        # Merge images in Fourier domain
        miriad.immerge(
            _in=f"{mir_im_self},{mir_im_other}",
            out=str(mir_im_merge),
            options='notaper', factor=1.0
        )

        miriad.fits(
            _in=str(mir_im_merge), out=str(fits_merge), op='xyout'
        )

        # Correct merged image pixel fluxes so total flux per channel in
        # merged image is the same as that of the low-resolution image, if
        # desired
        if normalise:
            lr_total_fluxes = np.nansum(other.data('JY/PIXEL'), axis=(1, 2))
            with fits.open(fits_merge) as hdul:
                factor = lr_total_fluxes - np.nansum(hdul[0].data, axis=(1, 2))
                factor /= np.product(np.shape(hdul[0].data)[-2:])
                hdul[0].data += factor[:, np.newaxis, np.newaxis]
                hdul.writeto(fits_merge, overwrite=True)

        # Clean up temporary miriad images
        for im in (mir_im_merge, mir_im_self, mir_im_other):
            shutil.rmtree(im)

        # Instantiate new SkyComponent instance to return
        new_sky_comp = SkyComponent.load_from_fits(fits_merge, new_name)

        # Clean up temporary fits images
        fits_merge.unlink()

        return new_sky_comp

    def t_b(self, freq: tbfs.FreqType) -> tbfs.ReturnType:
        """Brightness temperature distribution on the sky as an array [K]"""
        return self._tb_func(self, freq)


class SubbandSkyModel(_BaseSkyClass):
    """
    Class composed of multiple, individual SkyComponent instances to describe
    the sky's brightness distribution for a single Subband (i.e. contiguous
    set of frequencies)
    """
    from ..observing import Subband

    def __init__(self, n_pix: Tuple[int, int], cdelt: float, coord0: SkyCoord,
                 subband: Subband):
        """

        Parameters
        ----------
        n_pix
            Number of pixels in x (R.A.) and y (declination) as a 2-tuple
        cdelt
            Pixel size [deg]
        coord0
            Central coordinate (corresponds to the fits-header CRVAL1 and
            CRVAL2 keywords) as a astropy.coordinates.SkyCoord instance
        subband
            Subband corresponding to the SubbandSkyModel instance's desired
            sky brightness distribution cube's spectral axis
        """
        super().__init__(n_pix=n_pix, cdelt=cdelt, coord0=coord0)
        self._subband = subband
        self._tb_data = np.zeros((len(self.frequencies), self.n_y, self.n_x))
        self._components = []

    @property
    def frequencies(self) -> npt.NDArray:
        """Array of frequencies this SubbandSkyModel covers [Hz]"""
        return self._subband.frequencies

    @property
    def subband(self) -> Subband:
        """Subband instance this SubbandSkyModel corresponds to"""
        return self._subband

    def t_b(self, freq: float) -> np.ndarray:
        """
        Brightness temperature array at specified frequency (must be listed in
        self.frequencies)
        """
        idx = np.squeeze(np.isclose(self.frequencies, freq, atol=1.))

        return self._tb_data[idx]

    @property
    def components(self):
        """Constituent SkyComponent instances"""
        return self._components

    def __add__(self, other: Union[List[SkyComponent],
                                   Tuple[SkyComponent],
                                   SkyComponent]) -> 'SubbandSkyModel':
        """
        Magic __add__ method used for a convenient interface with the
        SubbandSkyModel.add_component method
        """
        self.add_component(other)
        return self

    def add_component(self, new_component: Union[List[SkyComponent],
                                                 Tuple[SkyComponent],
                                                 SkyComponent]):
        """
        Adds a SkyComponent instance to the SubbandSkyModel and adds its contribution
        to the SubbandSkyModel's brightness temperature distribution

        Parameters
        ----------
        new_component
            SkyComponent instance, or iterable containing SkyComponent instances

        Raises
        -------
        TypeError
            If one of components/the component being added is not an instance of
            the SkyComponent class or its children
        """
        from ..physics import astronomy as ast

        if isinstance(new_component, (list, tuple, np.ndarray)):
            for component in new_component:
                self.add_component(component)

        else:
            if not self.possess_similar_header(new_component):
                errh.raise_error(ValueError,
                                 f"{new_component} incompatible with sky model")

            self._components.append(new_component)

            # NOTE: Assumption of optically-thin SkyModel and added component
            if new_component.format == 'image':
                self._tb_data += ast.intensity_to_tb(
                    new_component.data(unit='JY/SR'), new_component.frequencies
                )
            else:
                self._add_osm_to_table(new_component.oskar_table)

    @decorators.convert_str_to_path('osmfile')
    def _add_osm_to_table(self, osmfile: pathlib.Path):
        # dataframe_to_osm, osm_to_dataframe
        from ..miscellaneous import file_handling as fh

        df_to_append = fh.osm_to_dataframe(osmfile)

        if self.oskar_table is None:
            fname = pathlib.Path(
                f"{self.__class__.__name__}_Subband{self._subband.name}.osm"
            )

            fname.unlink(missing_ok=True)
            self.oskar_table = fname
            new_df = df_to_append

        else:
            current_df = fh.osm_to_dataframe(self.oskar_table)
            new_df = current_df.append(df_to_append)

        self.oskar_table.unlink(missing_ok=True)
        fh.dataframe_to_osm(new_df, self.oskar_table)

    def add_frequency(self, *args, **kwargs):
        """
        Raises
        ------
        NotImplementedError
            Since SubbandSkyModel frequencies should only be assigned to a SubbandSkyModel
            instance upon its creation, this error is raised to avoid user
            issues arising from misuse of the inherited
            _BaseSkyClass.add_frequency method
        """
        raise NotImplementedError(
            "SubbandSkyModel frequencies can only be defined upon creation of "
            "the SubbandSkyModel instance via the 'subband' arg"
        )

    def summary(self):
        """Return a handy summary as a str and also prints it"""
        from collections import Counter

        comp_fmt_counts = Counter([comp.format for comp in self.components])
        n_tabular = comp_fmt_counts['table']
        n_image = comp_fmt_counts['image']

        smry = (f"{self.__class__.__name__} instance, comprised of "
                f"{len(self.components)} components ({n_tabular} tabular fmt, "
                f"{n_image} image fmt) for the {self.subband.name} subband.")

        if n_tabular > 0:
            from ..miscellaneous.file_handling import osm_to_dataframe

            df = osm_to_dataframe(self.oskar_table)
            n_points = np.nansum((df.maj == 0).values |
                                 (df['min'] == 0).values)
            n_gaussians = np.nansum((df.maj != 0).values &
                                    (df['min'] != 0).values)

            smry += (f"\nTabular sources -> {n_gaussians} Gaussians, "
                     f"{n_points} point-like sources ({len(df)} total) "
                     f"written to {self.oskar_table.resolve()}")

        smry += (f"\nImage grid -> "
                 f"{self.n_x}\u00D7{self.n_y}, {self.cdelt * 3600:.1f} "
                 f"arcsec pixels")

        print(smry)

        return smry


class SkyModel:
    """
    Class composed of multiple, individual SubbandSkyModel instances to describe
    the sky's brightness distribution across multiple Subbands
    """
    from ..observing import Subband

    def __init__(self, n_pix: Tuple[int, int], cdelt: float, coord0: SkyCoord,
                 subbands: Optional[Union[Iterable[Subband], Subband]] = None):
        self.n_x, self.n_y = n_pix
        self.cdelt = cdelt
        self.coord0 = coord0
        self._subbands = []
        self._subband_skymodels = {}

        if subbands is not None:
            self.add_subband(subbands)

    @property
    def subbands(self):
        return self._subbands

    def add_subband(self, new_subband: Union[Iterable[Subband], Subband]):
        """
        Add subband(s) to this SkyModel's list of subbands and as a key to
        SkyModel's subband_skymodels
        """
        import collections.abc

        if isinstance(new_subband, collections.abc.Iterable):
            for subband in new_subband:
                self.add_subband(subband)

        else:
            self._subbands.append(new_subband)
            self._subband_skymodels[new_subband] = None

    @property
    def subband_skymodels(self):
        """Dict of Subband/SubbandSkyModel key/value pairs"""
        return self._subband_skymodels

    def is_not_matching_field(self, other: SubbandSkyModel
                              ) -> Union[ValueError, None]:
        """Check if field (n_x, n_y, cdelt) does NOT match a SubbandSkyModel"""
        attrs = ('n_x', 'n_y', 'cdelt')
        field_inconsistencies = [self.n_x != other.n_x,
                                 self.n_y != other.n_y,
                                 not np.isclose(self.cdelt, other.cdelt)]

        if any(field_inconsistencies):
            attr = attrs[field_inconsistencies.index(True)]
            msg = (f"Inconsistency between SkyModel and SubbandSkyModel {attr}"
                   f" -> SkyModel.{attr} != SubbandSkyModel.{attr}, "
                   f"{getattr(self, attr)} != {getattr(other, attr)}")

            return ValueError(msg)

    def add_subband_skymodel(self, subband: Subband, skymodel: SubbandSkyModel):
        """Add a SubbandSkyModel for a single Subband of the SkyModel"""
        if not all(np.isclose(subband.frequencies, skymodel.frequencies)):
            raise ValueError(
                f"Frequencies of subband, {subband} and added SubbandSkyModel "
                f"don't match"
            )

        if self.is_not_matching_field(skymodel):
            raise self.is_not_matching_field(skymodel)

        if subband not in self._subbands:
            self.add_subband(subband)

        self._subband_skymodels[subband] = skymodel