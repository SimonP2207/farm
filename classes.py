"""
All classes for use within the FARM infrastructure.
"""
import copy
import shutil
import pathlib
import tempfile
import math
import random
from abc import ABC, abstractmethod
from typing import Tuple, List, Union, TypeVar

import numpy.typing as npt
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io.fits import Header
from reproject import reproject_from_healpix
from pygdsm import GlobalSkyModel2016

from farm import LOGGER, DATA_FILES
from farm import decorators
from . import error_handling as errh
from . import astronomy as ast
from .software import miriad


SkyModelType = TypeVar('SkyModelType', bound='_BaseSkyClass')


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
        # # If average value is above T_CMB, assume it must be T_B
        # if np.nanmean(image_data) > 2.:
        #     return 'K'
        #
        # # Otherwise, it must be a flux or intensity
        # # Assume if there is beam information, then it is Jy/beam
        # if 'BMAJ' in header:
        #     return 'JY/BEAM'
        #
        # # Otherwise assume Jy/pixel
        # return 'JY/PIXEL'


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


def generate_random_chars(length: int, choices: str = 'alphanumeric') -> str:
    """
    For generating sequence of random characters for e.g. file naming

    Parameters
    ----------
    length
        Number of characters to generate
    choices
        Characters to choose from. Can be 'alphanumeric', 'alpha', or 'numeric.
        Default is 'alphanumeric
    Returns
    -------
    string of defined length comprised of random characters from desired
    character range

    Raises
    ------
    ValueError
        If 'choices' not one of 'alphanumeric', 'alpha' or 'numeric'
    """
    if choices not in ('alphanumeric', 'alpha', 'numeric'):
        raise ValueError("choices must be one of 'alphanumeric', 'alpha', or "
                         f"'numeric', not {choices}")
    poss_chars = ''
    if 'alpha' in choices:
        poss_chars += ''.join([chr(_) for _ in range(65, 91)])
        poss_chars += ''.join([chr(_) for _ in range(97, 123)])
    if 'numeric' in choices:
        poss_chars += ''.join([chr(_) for _ in range(49, 58)])

    assert poss_chars, "Not sure how poss_chars is an empty string..."

    return ''.join([random.choice(poss_chars) for _ in range(length)])


def hdr2d_from_skymodel(sky_class: SkyModelType) -> Header:
    hdr_dict = {
        'BITPIX': -32,  # Assuming all pixel values in range -3.4E38 to +3.4E38
        'NAXIS': 2,
        'NAXIS1': sky_class.n_x,
        'NAXIS2': sky_class.n_y,
        'CTYPE1': 'RA---SIN',
        'CTYPE2': 'DEC--SIN',
        'CRVAL1': sky_class.coord0.ra.deg,
        'CRVAL2': sky_class.coord0.dec.deg,
        'CRPIX1': sky_class.n_x / 2,
        'CRPIX2': sky_class.n_y / 2,
        'CDELT1': -sky_class.cdelt,
        'CDELT2': sky_class.cdelt,
        'CUNIT1': 'deg     ',
        'CUNIT2': 'deg     ',
        'EQUINOX': {'fk4': 1950., 'fk5': 2000.}[sky_class.coord0.frame.name],
    }

    # Guarantee order in which keywords are added to .fits header
    order = ('BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2',
             'CTYPE1', 'CTYPE2', 'CRVAL1', 'CRVAL2',
             'CRPIX1', 'CRPIX2', 'CDELT1', 'CDELT2',
             'CUNIT1', 'CUNIT2', 'EQUINOX')

    hdr = Header({'Simple': True})
    for keyword, value in ((kw, hdr_dict[kw]) for kw in order):
        hdr.set(keyword, value)

    return hdr


def hdr3d_from_skyclass(sky_class: SkyModelType) -> Header:
    if len(sky_class.frequencies) == 0:
        raise ValueError("Can't create Header from SkyClass with no frequency "
                         "information")

    hdr_dict = {
        'NAXIS3': len(sky_class.frequencies),
        'CTYPE3': 'FREQ    ',
        'CRVAL3': min(sky_class.frequencies),
        'CRPIX3': 1,
        'CDELT3': sky_class.frequencies[1] - sky_class.frequencies[0]
        if len(sky_class.frequencies) > 1 else 1,
        'CUNIT3': 'Hz      ',
    }

    hdr = hdr2d_from_skymodel(sky_class)
    hdr.insert('NAXIS2', ('NAXIS3', None), after=True)
    hdr.insert('CTYPE2', ('CTYPE3', 'FREQ    '), after=True)
    hdr.insert('CRVAL2', ('CRVAL3', None), after=True)
    hdr.insert('CRPIX2', ('CRPIX3', 1), after=True)
    hdr.insert('CDELT2', ('CDELT3', None), after=True)
    hdr.insert('CUNIT2', ('CUNIT3', 'Hz      '), after=True)

    order = ('NAXIS3', 'CTYPE3', 'CRVAL3', 'CRPIX3', 'CDELT3', 'CUNIT3')

    for keyword, value in ((kw, hdr_dict[kw]) for kw in order):
        hdr.set(keyword, value)

    return hdr


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
    @decorators.docstring_parameter(str(VALID_UNITS)[1:-1])
    def load_from_fits(cls, fitsfile: pathlib.Path) -> 'SkyComponent':
        """
        Creates a SkyComponent instance from a .fits cube

        Parameters
        ----------
        fitsfile
            Full path to .fits file

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
        LOGGER.info(f"Loading {str(fitsfile.resolve())}")
        if not fitsfile.exists():
            errh.raise_error(FileNotFoundError,
                             f"{str(fitsfile.resolve())} not found")

        hdr, data = fits_hdr_and_data(fitsfile)

        if data.ndim != 3:
            errh.raise_error(ValueError, ".fits must be a cube (3D), but has "
                                         f"{data.ndim} dimensions")

        # Image information
        nx_, ny_, cdelt_ = hdr["NAXIS1"], hdr["NAXIS2"], hdr["CDELT2"]
        ra_cr, dec_cr = hdr["CRVAL1"], hdr["CRVAL2"]
        freqs = fits_frequencies(fitsfile)  # Spectral axis information
        equinox = fits_equinox(fitsfile)  # 1950.0 or 2000.0
        unit = fits_bunit(fitsfile)  # Brightness unit information

        if not unit:
            raise ValueError("No brightness unit information present in "
                             f"{str(fitsfile)}")

        if unit not in cls.VALID_UNITS:
            raise ValueError("Please ensure unit is one of "
                             f"{repr(cls.VALID_UNITS)[1:-1]}, not "
                             f"'{unit}'")

        frame = {1950.0: 'fk4', 2000.0: 'fk5'}[equinox]
        coord0 = SkyCoord(ra_cr, dec_cr, unit=(u.degree, u.degree),
                          frame=frame)

        if unit not in cls.VALID_UNITS:
            errh.raise_error(ValueError,
                             f"Something has gone unexpectedly wrong. "
                             f"Unit in loaded .fits image is '{unit}'")
        else:
            if unit == 'K':
                pass
            elif unit == 'JY/SR':
                data = ast.intensity_to_tb(data, freqs)
            elif unit == 'JY/PIXEL':
                data = ast.flux_to_tb(data, freqs, np.radians(cdelt_) ** 2.)

        sky_model = SkyComponent((nx_, ny_), cdelt=cdelt_, coord0=coord0)
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
    def frequencies(self) -> npt.ArrayLike:
        return self._frequencies

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
        temp_fits_file = pathlib.Path(f"{temp_pfx}_"
                                      f"temp_{generate_random_chars(10)}.fits")

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

    def regrid(self, template: SkyModelType) -> SkyModelType:
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
        out_fits_image = pathlib.Path(f'out_image_{sffx}.fits')

        temporary_images = (template_mir_image, input_mir_image,
                            out_mir_image, out_fits_image)

        for temporary_image in temporary_images:
            if temporary_image.exists():
                shutil.rmtree(temporary_image)

        template.write_miriad_image(template_mir_image, unit='K')
        self.write_miriad_image(input_mir_image, unit='K')

        miriad.regrid(_in=input_mir_image, tin=template_mir_image,
                      out=out_mir_image)
        miriad.fits(op='xyout', _in=out_mir_image, out=out_fits_image)

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

    def possess_similar_header(self, other: SkyModelType) -> bool:
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

    def same_spectral_setup(self, other: SkyModelType) -> bool:
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

        if not all(self.frequencies == other.frequencies):
            return False

        return True


class SkyComponent(_BaseSkyClass):
    def __init__(self, n_pix: Tuple[int, int], cdelt: float, coord0: SkyCoord,
                 model: Union[None, str] = None):
        super().__init__(n_pix, cdelt, coord0)
        self._model = model

    @property
    def model(self):
        return self._model

    def normalise(self, other: SkyModelType,
                  inplace: bool = False) -> Union[None, SkyModelType]:
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
        None
        """
        self_sum_td_nu = np.nansum(self.data(unit='JY/SR'), axis=(1, 2))
        other_sum_td_nu = np.nansum(other.data(unit='JY/SR'), axis=(1, 2))

        scalings = (other_sum_td_nu / self_sum_td_nu)[:, np.newaxis, np.newaxis]

        if inplace:
            self._tb_data *= scalings

        else:
            new_skymodeltype = copy.deepcopy(self)
            new_skymodeltype._tb_data *= scalings
            return new_skymodeltype

    def t_b(self, freq: Union[float, npt.ArrayLike]) -> np.ndarray:
        raise NotImplementedError


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
        # Attributes created from constructor args
        self.n_x, self.n_y = n_pix
        self.cdelt = cdelt
        self.coord0 = coord0
        self._frequencies = frequencies

        # Private instance attributes
        self._tb_data = np.zeros((len(self.frequencies), self.n_y, self.n_x))
        self._components = []

    def t_b(self, freq: float) -> np.ndarray:
        idx = np.squeeze(np.isclose(self.frequencies, freq, atol=1.))

        return self._tb_data[idx]

    @property
    def components(self):
        return self._components

    @property
    def frequencies(self):
        return self._frequencies

    @frequencies.setter
    def frequencies(self, *args, **kwargs):
        pass

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


class GDSM(SkyComponent):
    """
    Diffuse, large-scale sky model for Galactic emission

    Class Attributes
    ----------
    VALID_MODELS : tuple of str
        Models valid to instantiate a DiffuseSkyModel instance with

    Attributes
    ----------
    model : str
        Model used to instantiate the instance and calculate the brightness
        temperature distribution
    """
    VALID_MODELS = ('GSM2016',)  # Valid Galactic sky models

    def __init__(self, npix: Tuple[int, int], cdelt: float, coord0: SkyCoord,
                 model: str = 'GSM2016'):
        """
        Extends the behaviour of superclass constructor method by requiring a
        model for the Galactic diffuse emission

        Parameters
        ----------
        npix
            Number if pixels in (x, y) of the SkyModel
        cdelt
            Pixel size [deg]
        coord0
            Central coordinate of SkyModel
        model
            Model with which to calculate brightness temperature distribution.
            See class variable 'VALID_MODELS' for accepted values. Default is
            'GSM2016'

        Raises
        ------
        ValueError
            When 'model' value is not in DiffuseSkyModel.VALID_MODELS
        """
        if model not in self.VALID_MODELS:
            errh.raise_error(ValueError,
                             f"{model} not a valid diffuse model")

        super().__init__(npix, cdelt, coord0, model)

    def t_b(self, freq: float) -> np.ndarray:
        gdsm = None
        if self.model == 'GSM2016':
            gdsm = GlobalSkyModel2016(freq_unit='Hz', data_unit='MJysr',
                                      resolution='hi')
        else:
            errh.raise_error(ValueError,
                             f"{self.model} is not a recognised model for "
                             f"DiffuseSkyModel class. This should have been "
                             f"caught within the __init__ constructor method?")

        temp_fitsfile = pathlib.Path(f'temp{generate_random_chars(10)}.fits')

        gdsm.generate(freq)
        gdsm.write_fits(str(temp_fitsfile))  # expects str type

        hdugsm = fits.open(temp_fitsfile)
        hdugsm[1].header.set('COORDSYS', 'G')

        temp_fitsfile.unlink()

        i_nu = np.single(reproject_from_healpix(hdugsm[1], self.header2d))[0]
        i_nu *= 1e6  # MJy/sr -> Jy/sr

        return ast.intensity_to_tb(i_nu, freq)


class GSSM(SkyComponent):
    """
    Small-scale (e.g. filamentary) sky model for Galactic emission

    Class Attributes
    ----------
    VALID_MODELS : tuple of str
        Models valid to instantiate a SmallSkyModel instance with

    Attributes
    ----------
    model : str
        Model used to instantiate the instance and calculate the brightness
        temperature distribution
    """
    VALID_MODELS = ('MHD',)  # Valid Galactic sky models

    @decorators.docstring_parameter(str(VALID_MODELS)[1:-1])
    def __init__(self, npix: Tuple[int, int], cdelt: float, coord0: SkyCoord,
                 model: str = 'MHD'):
        """
        Extends the behaviour of superclass constructor method by requiring a
        model for the Galactic small_scale emission

        Parameters
        ----------
        npix
            Number of pixels in (x, y), or (ra, dec)
        cdelt
            Pixel size [deg]
        coord0
            Central coordinate of SkyModel
        model
            Model with which to calculate brightness temperature distribution.
            See class variable 'VALID_MODELS' for accepted values. Default is
            'MHD', for which a new GSSM instance is instantiated

        Raises
        ------
        ValueError
            When 'model' arg is not one of {0}
        """
        if model not in self.VALID_MODELS:
            errh.raise_error(ValueError,
                             f"{model} not a valid short-scale model")

        if model == 'MHD':
            sm = self.load_from_fits(DATA_FILES["MHD"])
            if npix != (sm.n_x, sm.n_y):
                errh.issue_warning(UserWarning,
                                   "For MHD GSSM model, must adopt image size "
                                   f"of {sm.n_x} x {sm.n_y}. Ignoring request "
                                   f"for {npix[0]} x {npix[1]}")

            super().__init__((sm.n_x, sm.n_y), cdelt, coord0,)
            self._frequencies = sm.frequencies
            self._tb_data = sm.data(unit='K')

        self._model = model

    def t_b(self, freq: float) -> np.ndarray:
        if self.model == 'MHD':
            return self.data('K')[np.where(self.frequencies == freq)[0][0]]
