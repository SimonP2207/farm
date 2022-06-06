"""
All classes for use within the FARM infrastructure.
"""
import copy
import shutil
import pathlib
import tempfile
import math
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Union, TypeVar, Type

import numpy.typing as npt
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io.fits import Header

from ..miscellaneous.fits import fits_table_to_dataframe
from ..data.loader import Correlator
from ..miscellaneous import decorators, error_handling as errh, interpolate_values, generate_random_chars
from ..physics import astronomy as ast
from ..software import miriad
from . import tb_functions as tbfs

# Typing related code
SkyClassType = TypeVar('SkyClassType', bound='_BaseSkyClass')


# Miscellaneous functions
# TODO: Relocate these to a sensible module
def fits_bunit(fitsfile: pathlib.Path) -> str:
    """
    Get brightness unit from .fits header. If 'BUNIT' not present in .fits
    header, best guess the data unit
    """
    header, image_data = fits_hdr_and_data(fitsfile)

    if 'BUNIT' in header:
        return header["BUNIT"].strip().upper()
    else:
        return None


def fits_equinox(fitsfile: pathlib.Path) -> float:
    """Get equinox from .fits header. Assume J2000 if absent"""
    header, _ = fits_hdr_and_data(fitsfile)

    try:
        return header["EQUINOX"]
    except KeyError:
        # Assume J2000 if information not present in header
        errh.issue_warning(UserWarning,
                           "Equinox information not present. Assuming J2000")
        return 2000.0


def fits_hdr_and_data(fitsfile: pathlib.Path) -> Tuple[Header, np.ndarray]:
    """Return header and data from a .fits image/cube"""
    with fits.open(fitsfile) as hdulist:
        return hdulist[0].header, hdulist[0].data


def fits_frequencies(fitsfile: pathlib.Path) -> np.ndarray:
    """Get list of frequencies of a .fits cube, as np.ndarray"""
    header, _ = fits_hdr_and_data(fitsfile)
    return fits_hdr_frequencies(header)


def fits_hdr_frequencies(header: Header) -> np.ndarray:
    """Get list of frequencies from a .fits header, as np.ndarray"""
    freq_min = (header["CRVAL3"] - (header["CRPIX3"] - 1) *
                header["CDELT3"])
    freq_max = (header["CRVAL3"] + (header["NAXIS3"] - header["CRPIX3"]) *
                header["CDELT3"])
    return np.linspace(freq_min, freq_max, header["NAXIS3"])


def hdr2d(n_x: int, n_y: int, coord0, cdelt, frame='fk5') -> Header:
    hdr = Header({'Simple': True})
    hdr.set('BITPIX', -32)
    hdr.set('NAXIS', 2)
    hdr.set('NAXIS1', n_x)
    hdr.set('NAXIS2', n_y)
    hdr.set('CTYPE1', 'RA---SIN')
    hdr.set('CTYPE2', 'DEC--SIN')
    hdr.set('CRVAL1', coord0.ra.deg)
    hdr.set('CRVAL2', coord0.dec.deg)
    hdr.set('CRPIX1', n_x / 2)
    hdr.set('CRPIX2', n_y / 2)
    hdr.set('CDELT1', -cdelt)
    hdr.set('CDELT2', cdelt)
    hdr.set('CUNIT1', 'deg     ')
    hdr.set('CUNIT2', 'deg     ')
    hdr.set('EQUINOX', {'fk4': 1950., 'fk5': 2000.}[coord0.frame.name])

    return hdr


def hdr3d(n_x: int, n_y: int, coord0, cdelt, frequencies: npt.ArrayLike,
          frame='fk5') -> Header:

    hdr = hdr2d(n_x, n_y, coord0, cdelt, frame)
    hdr.insert('NAXIS2', ('NAXIS3', len(frequencies)), after=True)
    hdr.insert('CTYPE2', ('CTYPE3', 'FREQ    '), after=True)
    hdr.insert('CRVAL2', ('CRVAL3', min(frequencies)), after=True)
    hdr.insert('CRPIX2', ('CRPIX3', 1), after=True)
    hdr.insert('CDELT2', ('CDELT3', 1.), after=True)
    hdr.insert('CUNIT2', ('CUNIT3', 'Hz      '), after=True)

    if len(frequencies) > 1:
        hdr.set('CDELT3', frequencies[1] - frequencies[0])

    return hdr


def deconvolve_cube(input_fits, output_fits, beam):
    """Deconvolves a .fits image with a beam. DO NOT USE: Unstable results"""
    from ..software.miriad import miriad

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
    return hdr2d(sky_class.n_x, sky_class.n_y, sky_class.coord0,
                 sky_class.cdelt, sky_class.coord0.frame.name)


def hdr3d_from_skyclass(sky_class: Type[SkyClassType]) -> Header:
    if len(sky_class.frequencies) < 1:
        raise ValueError("Can't create Header from SkyClass with no frequency "
                         "information")

    return hdr3d(sky_class.n_x, sky_class.n_y, sky_class.coord0,
                 sky_class.cdelt, sky_class.frequencies,
                 sky_class.coord0.frame.name)


def deconvolve_fwhm(conv_size: float, beam_size: float) -> float:
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
    _FREQ_TOL = 1.
    VALID_UNITS = ('K', 'JY/PIXEL', 'JY/SR')

    @classmethod
    def load_from_fits_table(cls, columns: dict,
                             fitsfile: Union[pathlib.Path, fits.HDUList],
                             name: str, cdelt: float,
                             coord0: SkyCoord, fov: Tuple[float, float],
                             freqs: npt.ArrayLike,
                             flux_range: Tuple[float, float] = [0., 1e30],
                             beam: Optional[dict] = None) -> 'SkyComponent':
        """
        Creates a SkyComponent instance from a .fits table file or HDUList
        loaded from such

        Parameters
        ----------
        columns
            Dictionary of columns whose keys are 'ra', 'dec', 'fluxI', 'freq0',
            'spix', 'maj', 'min', and 'pa' and corresponding values are the
            column names corresponding to those parameters within the fits
            table. For example:
                {'ra': 'RAJ2000', 'dec': 'DEJ2000', 'fluxI': 'int_flux_wide',
                 'freq0': 'freq0', 'spix': 'alpha', 'maj': 'a_wide',
                 'min': 'b_wide', 'pa': 'pa_wide'}
        fitsfile
            pathlib.Path to .fits table, or HDUList instance
        name
            Name to give returned SkyComponent instance
        cdelt
            Cells size [deg]
        coord0
            Central coordinate
        fov
            Field of view extent in x and y as a tuple [deg]
        freqs
            Frequencies of the SkyComponent
        flux_range
            Lower and upper bound of source fluxes as a 2-tuple, (lower, upper)
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
        from astropy import wcs
        from ..miscellaneous import image_functions as imfunc

        # Set up fits header, WCS and data array
        im_hdr = hdr3d(int(fov[0] // cdelt),
                       int(fov[1] // cdelt),
                       coord0, cdelt, freqs, 'fk5')
        im_hdr.insert('CUNIT3', ('BUNIT', 'JY/PIXEL'), after=True)
        im_wcs = wcs.WCS(im_hdr)
        im_data = np.zeros((im_hdr['NAXIS3'],
                            im_hdr['NAXIS2'],
                            im_hdr['NAXIS1']))

        # Set up pixel grid and coordinate grid
        zz, yy, xx = np.meshgrid(np.arange(im_hdr['NAXIS3']),
                                 np.arange(im_hdr['NAXIS2']),
                                 np.arange(im_hdr['NAXIS1']), indexing='ij')
        rra, ddec, ffreq = im_wcs.wcs_pix2world(xx, yy, zz, 0)

        if isinstance(fitsfile, pathlib.Path):
            data = fits_table_to_dataframe(fitsfile)
        elif isinstance(fitsfile, fits.HDUList):
            data = pd.DataFrame.from_records(fitsfile[1].data)
        else:
            errh.raise_error(TypeError, f"{fitsfile} not an HDUList instance "
                                        "or path to a .fits table")

        data[columns['spix']] = np.where(np.isnan(data.alpha), -0.7, data.alpha)
        fov_mask = ast.within_square_fov(
            fov, coord0.ra.deg, coord0.dec.deg,
            data[columns['ra']], data[columns['dec']]
        )

        flux_range_mask = ((data[columns['fluxI']] >= flux_range[0]) &
                           (data[columns['fluxI']] <= flux_range[1]))

        data = data[fov_mask & flux_range_mask]

        # Deconvolve given sizes from beam, if given
        if beam:
            if not np.isclose(beam['maj'], beam['min'], rtol=1e-2):
                errh.raise_error(ValueError,
                                 "Circular beams only for deconvolution of "
                                 "point source table ")

            data[columns['maj']] = deconvolve_fwhm(
                data[columns['maj']], beam['maj']
            )
            data[columns['min']] = deconvolve_fwhm(
                data[columns['min']], beam['min']
            )

            # Default size for sources smaller than the beam
            point_source_size = cdelt / 5.
            data[columns['maj']] = np.where(
                np.isnan(data[columns['maj']]),
                point_source_size, data[columns['maj']]
            )

            data[columns['min']] = np.where(
                np.isnan(data[columns['min']]),
                point_source_size, data[columns['min']]
            )

        # Peak intensity calculation in Jy/pixel
        data['peak_int'] = (
                data[columns['fluxI']] * np.sqrt(2. * np.log(2.)) /
                (np.pi * data[columns['maj']] *
                 data[columns['min']] * 2.350443e-11) *
                np.radians(cdelt) ** 2.
        )

        # Loop through all sources and add to grid
        for _, row in data.iterrows():
            idxs = im_wcs.world_to_array_index_values(
                row[columns['ra']], row[columns['dec']], freqs[0]
            )

            # Width in indices of sub-array within data to calculate source flux
            didx = int(row[columns['maj']] * 2 / 3600. // cdelt + 1)
            didx = np.max([didx, 5])

            # Index ranges in ra and dec within which to calculate source flux
            ra_idx = (np.max([idxs[2] - didx, 0]),
                      np.min([idxs[2] + didx, im_hdr['NAXIS1']]))

            dec_idx = (np.max([idxs[1] - didx, 0]),
                       np.min([idxs[1] + didx, im_hdr['NAXIS2']]))

            rra_ = rra[0, dec_idx[0]:dec_idx[1], ra_idx[0]:ra_idx[1]]
            ddec_ = ddec[0, dec_idx[0]:dec_idx[1], ra_idx[0]:ra_idx[1]]

            # If source if near RA = 0, imfunc.gaussian_2d is given coordinates
            # in rra_ that it calculates are ~360deg from source position (due
            # to the wrapped nature of RA, which imfunc.gaussian_2d is unaware
            # of) leading to zeroes. Therefore unwrap the rra_ coordinates if
            # required
            if np.ptp(rra_) > 180:
                ra0 = row[columns['ra']]
                rra_ = np.where(np.abs(rra_ - ra0) > 180,
                                rra_ + (360. if ra0 > 180 else -360.),
                                rra_)

            val0 = imfunc.gaussian_2d(rra_, ddec_,
                                      row[columns['ra']], row[columns['dec']],
                                      row['peak_int'],
                                      row[columns['maj']],
                                      row[columns['min']],
                                      row[columns['pa']])

            for freq_idx in range(len(freqs)):
                vals = val0 * (freqs[freq_idx] /
                               row[columns['freq0']]) ** row[columns['spix']]
                im_data[freq_idx,
                        dec_idx[0]:dec_idx[1],
                        ra_idx[0]:ra_idx[1]] += vals

        # Generate temporary .fits image to call load_from_fits method on
        hdu = fits.PrimaryHDU(data=im_data, header=im_hdr)
        temp_identifier = generate_random_chars(10)
        temp_fits_file = pathlib.Path(f"{name}_temp_{temp_identifier}.fits")
        hdu.writeto(temp_fits_file, overwrite=True)

        sky_comp = cls.load_from_fits(temp_fits_file, name, cdelt, coord0)
        temp_fits_file.unlink()

        return sky_comp

    @classmethod
    @decorators.docstring_parameter(str(VALID_UNITS)[1:-1])
    def load_from_fits(
        cls, fitsfile: pathlib.Path,
        name: Optional[str] = None,
        cdelt: Optional[float] = None,
        coord0: Optional[SkyCoord] = None,
        freqs: Optional[npt.ArrayLike] = None
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
                        raise ValueError("Desired frequencies go past .fits "
                                         "frequency coverage. Interpolation of"
                                         "flux values (not extrapolation) only "
                                         "supported")
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

        sky_model = SkyComponent(name, (nx, ny), cdelt=cdelt,
                                 coord0=coord0, tb_func=tbfs.fits_t_b)
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

    def add_frequency(self, new_freq: Union[float, npt.ArrayLike]):
        """
        Add an observing frequency (or frequencies)

        Parameters
        ----------
        new_freq
            Observing frequency to add. Can be a float or iterable of floats

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If one of frequencies/the frequency being added is not a float
        """
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
        return hdr3d_from_skyclass(self)

    @property
    def header2d(self):
        return hdr2d_from_skymodel(self)

    @abstractmethod
    def t_b(self, freq: Union[float, npt.ArrayLike]) -> np.ndarray:
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

    def i_nu(self, freq: Union[float, npt.ArrayLike]) -> np.ndarray:
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
        return ast.tb_to_intensity(self.t_b(freq), freq)

    def flux_nu(self, freq: Union[float, npt.ArrayLike]):
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
        solid_angle = np.radians(self.cdelt) ** 2.

        return ast.intensity_to_flux(self.i_nu(freq), solid_angle)

    @decorators.docstring_parameter(str(VALID_UNITS)[1:-1])
    def write_fits(self, fits_file: pathlib.Path, unit: str):
        """
        Write .fits cube of intensities or brightness temperatures

        Parameters
        ----------
        fits_file
            Full path to write .fits file to
        unit
            One of {0}

        Returns
        -------
        None

        Raises
        ------
        ValueError
            When supplied unit is not one of {0}
        """
        from .. import LOGGER

        LOGGER.info(f"Generating .fits file, {str(fits_file)}")

        if unit not in self.VALID_UNITS:
            err_msg = f"{unit} not a valid unit. Choose one of " \
                      f"{str(self.VALID_UNITS)[1:-1]}"
            errh.raise_error(ValueError, err_msg)

        gsmhdu = fits.PrimaryHDU(self.data(unit=unit))
        hdr = self.header
        hdr.set('BUNIT', format(unit, '8'))
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
        Check if frequency already present in SkyModel

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
            return False
        if self.n_y != other.n_y:
            return False
        if not all([other.frequency_present(f) for f in self.frequencies]):
            return False

        rel_pix_tol = 0.5 / max([self.n_x, self.n_y])
        if not math.isclose(self.cdelt, other.cdelt, rel_tol=rel_pix_tol):
            return False

        if not math.isclose(self.coord0.ra.deg, other.coord0.ra.deg,
                            rel_tol=rel_pix_tol):
            return False

        if not math.isclose(self.coord0.dec.deg, other.coord0.dec.deg,
                            rel_tol=rel_pix_tol):
            return False

        return True

    def same_spectral_setup(self,
                            other: Union[Type[SkyClassType], Correlator]) -> bool:
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
    def __init__(self, name: str, npix: Tuple[int, int], cdelt: float,
                 coord0: SkyCoord, tb_func: tbfs.TbFunction):
        super().__init__(npix, cdelt, coord0)
        self.name = name
        self._tb_func = tb_func

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
        return self._tb_func(self, freq)


class SkyModel(_BaseSkyClass):
    def __init__(self, n_pix: Tuple[int, int], cdelt: float, coord0: SkyCoord,
                 frequencies: npt.ArrayLike):
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
        frequencies
            Frequencies corresponding to the SkyModel instance's desired sky
            brightness distribution cube's spectral axis
        """
        super().__init__(n_pix=n_pix, cdelt=cdelt, coord0=coord0)

        self._frequencies = frequencies
        self._tb_data = np.zeros((len(self.frequencies), self.n_y, self.n_x))

        self._components = []

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
                                   npt.ArrayLike, SkyComponent]) -> 'SkyModel':
        """
        Magic __add__ method used for a convenient interface with the
        SkyModel.add_component method
        """
        self.add_component(other)
        return self

    def add_component(self, new_component: Union[List[SkyComponent],
                                                 Tuple[SkyComponent],
                                                 npt.ArrayLike, SkyComponent]):
        """
        Adds a SkyComponent instance to the SkyModel and adds its contribution
        to the SkyModel's brightness temperature distribution

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
        if isinstance(new_component, (list, tuple, np.ndarray)):
            for component in new_component:
                self.add_component(component)

        elif not isinstance(new_component, SkyComponent):
            raise TypeError("Only SkyComponent instances can be added to "
                            f"SkyModel, not a {type(new_component)} instance")

        else:
            if not self.possess_similar_header(new_component):
                errh.raise_error(ValueError,
                                 f"{new_component} incompatible with sky model")

            self._components.append(new_component)

            # NOTE: Assumption of optically-thin SkyModel and added component
            self._tb_data += ast.intensity_to_tb(
                new_component.data(unit='JY/SR'), new_component.frequencies
            )

    def add_frequency(self, *args, **kwargs):
        """
        Raises
        ------
        NotImplementedError
            Since SkyModel frequencies should only be assigned to a SkyModel
            instance upon its creation, this error is raised to avoid user
            issues arising from misuse of the inherited
            _BaseSkyClass.add_frequency method
        """
        raise NotImplementedError("SkyModel frequencies can only be defined "
                                  "upon creation of the SkyModel instance")
