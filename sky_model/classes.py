"""
All classes for use within the FARM infrastructure.
"""
import copy
import shutil
import pathlib
import tempfile
import math
from abc import ABC, abstractmethod
from typing import Tuple, List, Union, TypeVar

import numpy.typing as npt
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io.fits import Header

from miscellaneous import decorators, error_handling as errh
from farm import LOGGER
from farm import astronomy as ast
from farm import miscellaneous as misc
from farm import tb_functions as tbfs
from farm.software import miriad

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


def hdr2d_from_skymodel(sky_class: SkyClassType) -> Header:
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


def hdr3d_from_skyclass(sky_class: SkyClassType) -> Header:
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
    def load_from_fits(cls, fitsfile: pathlib.Path,
                       name: str = "", cdelt: float = None,
                       coord0: SkyCoord = None) -> 'SkyComponent':
        """
        Creates a SkyComponent instance from a .fits cube

        Parameters
        ----------
        fitsfile
            Full path to .fits file
        cdelt
            Pixel size [deg]. By default this is parsed from the .fits header.
            Note that specifying a value artifically changes pixel size i.e.
            fluxes (i.e. Jy/pixel) stay the same from the input .fits to the
            output t_b data
        coord0
            Central coordinate (corresponds to the fits-header CRVAL1 and
            CRVAL2 keywords) as a astropy.coordinates.SkyCoord instance. By
            default this is parsed from the .fits header
        name
            Name to assign to the output instance. If not given, the fits
            filename (without the file type) will be given by default

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

        # Image information
        nx, ny = hdr["NAXIS1"], hdr["NAXIS2"]

        if cdelt is None:
            cdelt = hdr["CDELT2"]

        if coord0 is None:
            frame = {1950.0: 'fk4', 2000.0: 'fk5'}[equinox]
            coord0 = SkyCoord(hdr["CRVAL1"], hdr["CRVAL2"], unit=(u.deg, u.deg),
                              frame=frame)

        if unit == 'K':
            pass
        elif unit == 'JY/SR':
            data = ast.intensity_to_tb(data, freqs)
        elif unit == 'JY/PIXEL':
            data = ast.flux_to_tb(data, freqs, np.radians(cdelt) ** 2.)

        if name == "":
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
        temp_identifier = misc.generate_random_chars(10)
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

    def regrid(self, template: SkyClassType) -> SkyClassType:
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

        sffx = misc.generate_random_chars(10)
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

    def rotate(self, angle: float,
               inplace: bool = True) -> Union[None, SkyClassType]:
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
        from ..image_functions import rotate_image
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

    def possess_similar_header(self, other: SkyClassType) -> bool:
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

    def same_spectral_setup(self, other: SkyClassType) -> bool:
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
    def __init__(self, name: str, npix: Tuple[int, int], cdelt: float,
                 coord0: SkyCoord, tb_func: tbfs.TbFunction):
        super().__init__(npix, cdelt, coord0)
        self.name = name
        self._tb_func = tb_func

    def normalise(self, other: SkyClassType,
                  inplace: bool = False) -> Union[None, SkyClassType]:
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
        self_sum_td_nu = np.nansum(self.data(unit='JY/SR'), axis=(1, 2))
        other_sum_td_nu = np.nansum(other.data(unit='JY/SR'), axis=(1, 2))

        scalings = (other_sum_td_nu / self_sum_td_nu)[:, np.newaxis, np.newaxis]

        if inplace:
            self._tb_data *= scalings

        else:
            new_skymodeltype = copy.deepcopy(self)
            new_skymodeltype._tb_data *= scalings
            return new_skymodeltype

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
