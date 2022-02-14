"""
All classes for use within the FARM infrastructure.
"""
import shutil
import pathlib
import math
import random
from abc import ABC, abstractmethod
from typing import Tuple, Sequence, Union
import numpy as np
import scipy.constants as con
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io.fits import Header
import astropy.units as u
from reproject import reproject_from_healpix
from pygdsm import GlobalSkyModel2016

from farm import LOGGER, DATA_FILES
import errorhandling as errh
from software.miriad import miriad


def _gaussian_beam_area(bmaj: float, bmin: float) -> float:
    """
    Area of a Gaussian beam [sr]

    Parameters
    ----------
    bmaj
        Beam major axis FWHM [deg]
    bmin
        Beam minor axis FWHM [deg]

    Returns
    -------
    Total area under 2D Gaussian
    """
    bmaj_rad = np.radians(bmaj)
    bmin_rad = np.radians(bmin)

    return np.pi * bmaj_rad * bmin_rad / (4. * np.log(2.))


def _guess_fits_bunit(header: Header, image_data: np.ndarray) -> str:
    """
    Guess the data unit for a .fits image e.g. when the BUNIT header keyword is
    missing from the .fits header
    """
    # If average value is above T_CMB, assume it must be brightness temperature
    if np.nanmean(image_data) > 2.:
        return 'K'

    # Otherwise, it must be a flux or intensity
    # Assume if there is beam information, then it is Jy/beam
    if 'BMAJ' in header:
        return 'JY/BEAM'

    # Otherwise assume Jy/pixel
    return 'JY/PIXEL'


def _generate_random_chars(length: int, choices: str = 'alphanumeric') -> str:
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
    """
    if choices not in ('alphanumeric', 'alpha', 'numeric'):
        raise ValueError("choices must be one of 'alphanumeric', 'alpha', or "
                         f"'numeric', not {choices}")
    poss_chars = ''
    if 'alpha' in choices:
        poss_chars += ''.join([chr(_) for _ in range(65, 91)])
        poss_chars += ''.join([chr(_) for _ in range(97, 123)])
    if 'numeric' in choices:
        poss_chars += ''.join([chr(_) for _ in range(49,58)])

    assert poss_chars, "Not sure how poss_chars is an empty string..."

    return ''.join([random.choice(poss_chars) for _ in range(length)])


class SkyModel(ABC):
    """
    Abstract superclass for all subclasses represnting sky models (i.e.
    distributions of flux on the celestial sphere)
    """
    _FREQ_TOL = 1.  # Tolerance (Hz) determining if two frequencies are the same
    _VALID_UNITS = ('K', 'JY/PIXEL', 'JY/BEAM')

    @staticmethod
    def load_from_fits(fitsfile: pathlib.Path) -> 'LoadedSkyModel':
        """
        Creates a LoadedSkyModel instance from a .fits cube

        Parameters
        ----------
        fitsfile
            Full path to .fits file
        Returns
        -------
        LoadedSkyModel instance
        """
        LOGGER.info(f"Loading {fitsfile.__str__()} as LoadedSkyModel instance")
        if not fitsfile.exists():
            errh.raise_error(FileNotFoundError,
                             f"{fitsfile.__str__()} not found")

        hdulist = fits.open(fitsfile)[0]
        hdr, data = hdulist.header, hdulist.data

        # Image information
        nx, ny = hdr["NAXIS1"], hdr["NAXIS2"]
        cdelt = hdr["CDELT2"]

        # Spectral axis information
        freq_min = hdr["CRVAL3"] - (hdr["CRPIX3"] - 1) * hdr["CDELT3"]
        freq_max = hdr["CRVAL3"] + (hdr["NAXIS3"] -
                                    hdr["CRPIX3"]) * hdr["CDELT3"]
        freqs = np.linspace(freq_min, freq_max, hdr["NAXIS3"])

        # Coordinate information
        try:
            equinox = hdr["EQUINOX"]
        except KeyError:
            # Assume J2000 if information not present in header
            errh.issue_warning(UserWarning,
                               "Equinox information not present in "
                               f"{fitsfile.__str__()}, assuming J2000")
            equinox = 2000.0

        frame = {2.00E3: 'fk5', 1.95E3: 'fk4'}[equinox]
        coord0 = SkyCoord(hdr["CRVAL1"], hdr["CRVAL2"],
                          unit=(u.degree, u.degree), frame=frame)

        # Unit information
        if 'BUNIT' in hdr:
            unit = hdr["BUNIT"].strip().upper()
            if unit not in SkyModel._VALID_UNITS:
                errh.raise_error(ValueError, f"Unrecognised units, {unit} in "
                                             f"{fitsfile.__str__()}")
        else:
            unit = _guess_fits_bunit(hdr, data)

        if unit == 'K':
            pass
        elif unit in ('JY/PIXEL', 'JY/BEAM'):
            if unit == 'JY/PIXEL':
                solid_angle = (cdelt / 180. * con.pi) ** 2.
            elif unit == 'JY/BEAM':
                solid_angle = _gaussian_beam_area(hdr["BMAJ"], hdr["BMIN"])
            wavelengths = con.c / freqs
            conversion_to_k = wavelengths ** 2. * 1e-26 / \
                              (2. * con.Boltzmann * solid_angle)
            data *= conversion_to_k[:, np.newaxis, np.newaxis]
        else:
            errh.raise_error(ValueError,
                             f"Something has gone horribly wrong. "
                             f"Unit in loaded .fits is '{unit}'")

        sky_model = LoadedSkyModel((nx, ny), cdelt=cdelt, coord0=coord0,
                                   tb_data=data)
        sky_model.frequencies = list(freqs)
        _ = sky_model.hdr3d  # Generate header by calling hdr3d attribute

        # Add any missing .fits Header keywords that were missed during
        # instantiation of the LoadedSkyModel instance
        for keyword in hdr:
            if keyword not in sky_model.hdr3d:
                if keyword == 'HISTORY':
                    for line in str(hdr[keyword]).split('\n'):
                        if line:
                            sky_model.hdr3d.add_history(line)
                elif keyword == 'COMMENT':
                    for line in str(hdr[keyword]).split('\n'):
                        if line:
                            sky_model.hdr3d.add_comment(line)
                else:
                    try:
                        sky_model.hdr3d.set(keyword, hdr[keyword])
                    except ValueError:
                        errh.issue_warning(UserWarning,
                                           f"{keyword} not a recognised .fits "
                                           "header keyword and will not be "
                                           "added to the LoadedSkyModel "
                                           "instance's header")

        return sky_model

    def __init__(self, npix: Tuple[int, int], cdelt: float,
                 coord0: SkyCoord):
        """
        Parameters
        ----------
        npix
            Number if pixels in (x, y) of the SkyModel
        cdelt
            Pixel size [deg]
        coord0
            Central coordinate of SkyModel
        """
        self.n_x, self.n_y = npix
        self.cdelt = cdelt
        self.coord0 = coord0
        self._freqs = []
        self._hdr2d = None
        self._hdr3d = None

    @property
    def frequencies(self):
        """Sky model observing frequencies"""
        return self._freqs

    @frequencies.setter
    def frequencies(self, new_frequencies):
        self._freqs = new_frequencies

    def add_frequency(self, new_freq: Union[float, np.ndarray, Sequence[float]]):
        """
        Add an observing frequency (or frequencies) to the SkyModel

        Parameters
        ----------
        new_freq
            Observing frequency to add. Can be a float or iterable of floats
        Returns
        -------
        None
        """
        if isinstance(new_freq, (list, tuple, np.ndarray)):
            for freq in new_freq:
                self.add_frequency(freq)
        elif isinstance(new_freq, (float, np.floating)):
            dupl_freq = self.frequency_present(new_freq)
            if not dupl_freq:
                self.frequencies.append(new_freq)
                self.frequencies.sort()
            else:
                wrng_msg = f"Supplied frequency, {new_freq:.3f}Hz, already " \
                           f"present in SkyModel ({dupl_freq:.3f}Hz)"
                errh.issue_warning(UserWarning, wrng_msg)
        else:
            err_msg = f"Can't add a {type(new_freq)} to SkyModel frequencies," \
                      f" must be a float or list-like object containing floats"
            errh.raise_error(TypeError, err_msg)

    @property
    def hdr2d(self) -> Header:
        """
        FITS Header object for a 2D SkyModel image (i.e. no spectral axis)

        Returns
        -------
        astropy.io.fits.header.Header instance
        """
        if self._hdr2d is None:
            hdr2d = Header({'Simple': True})
            hdr2d.set('BITPIX', -32)
            hdr2d.set('NAXIS', 2)
            hdr2d.set('NAXIS1', self.n_x)
            hdr2d.set('NAXIS2', self.n_y)
            hdr2d.set('CTYPE1', 'RA---SIN')
            hdr2d.set('CTYPE2', 'DEC--SIN')
            hdr2d.set('CRVAL1', self.coord0.ra.deg)
            hdr2d.set('CRVAL2', self.coord0.dec.deg)
            hdr2d.set('CRPIX1', self.n_x / 2)
            hdr2d.set('CRPIX2', self.n_y / 2)
            hdr2d.set('CDELT1', -self.cdelt)
            hdr2d.set('CDELT2', self.cdelt)
            hdr2d.set('CUNIT1', 'deg     ')
            hdr2d.set('CUNIT2', 'deg     ')
            hdr2d.set('EQUINOX', {'fk5': 2000.,
                                  'fk4': 1950.}[self.coord0.frame.name])
            self._hdr2d = hdr2d

        return self._hdr2d

    @hdr2d.setter
    def hdr2d(self, new_hdr2d: Header):
        if not isinstance(new_hdr2d, Header):
            err_msg = f"Trying to assign as {type(new_hdr2d)} as .fits header"
            errh.raise_error(TypeError, err_msg)
        self._hdr2d = new_hdr2d

    @property
    def hdr3d(self) -> Union[Header, None]:
        """Get astropy.io.fits.Header instance for SkyModel"""
        if len(self.frequencies) == 0:
            wrng_msg = "Cannot assign header to SkyModel, add frequencies"
            errh.issue_warning(UserWarning, wrng_msg)
            return None

        if self._hdr3d is None:
            hdr3d = self.hdr2d.copy()
            hdr3d.set('NAXIS', 3)
            hdr3d.insert('CTYPE2', ('CTYPE3', 'FREQ    '), after=True)
            hdr3d.insert('CRPIX2', ('CRPIX3', 1), after=True)
            hdr3d.insert('CUNIT2', ('CUNIT3', 'Hz      '), after=True)
            hdr3d.insert('NAXIS2', ('NAXIS3', None), after=True)
            hdr3d.insert('CRVAL2', ('CRVAL3', None), after=True)
            hdr3d.insert('CDELT2', ('CDELT3', None), after=True)
            self._hdr3d = hdr3d

        self._hdr3d.set('NAXIS3', len(self.frequencies))
        self._hdr3d.set('CRVAL3', min(self.frequencies))
        # TODO: Probably should leave it to the user to set channel width
        if len(self.frequencies) > 1:
            self._hdr3d.set('CDELT3', self.frequencies[1] - self.frequencies[0])
        else:
            self._hdr3d.set('CDELT3', 1.)
            # TODO: rbraun set beam parameters...
            # hdr3d.header.set('BMAJ', bmaj)
            # hdr3d.header.set('BMIN', bmin)
            # hdr3d.header.set('BPA', bpa)

        return self._hdr3d

    @hdr3d.setter
    def hdr3d(self, new_hdr3d: Header):
        """Set SkyModel astropy.io.fits.Header or one of its values"""
        if not isinstance(new_hdr3d, Header):
            err_msg = f"Can't assign {type(new_hdr3d)} instance as .fits header"
            errh.raise_error(TypeError, err_msg)

        if self._hdr3d is None:
            err_msg = ".fits header hasn't been created and therefore can't " \
                      "be modified"
            errh.raise_error(TypeError, err_msg)

        self._hdr3d = new_hdr3d

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
        None
        """
        for freq_1 in self.frequencies:
            if math.isclose(freq, freq_1, abs_tol=abs_tol):
                return freq_1

        return False

    def generate_fits(self, fitsfile: pathlib.Path, unit: str = 'JY/PIXEL'):
        """
        Write .fits file cube of intensities or brightness temperatures

        Parameters
        ----------
        fitsfile
            Full path to write .fits file to
        unit
            One of either 'JY/PIXEL' (default) or 'K' for intensity or
            brightness temperature, respectively
        Returns
        -------
        None
        """
        LOGGER.info(f"Generating .fits file, {str(fitsfile)}")

        if unit not in self._VALID_UNITS:
            err_msg = f"{unit} not a valid unit. Choose one of " \
                      f"{', '.join(self._VALID_UNITS[:-1])} or " \
                      f"{self._VALID_UNITS[-1]}"
            errh.raise_error(ValueError, err_msg)

        gsmcube = np.empty((len(self.frequencies), self.n_y, self.n_x),
                           dtype=np.float32)
        for ichan, frequency in enumerate(self.frequencies):
            if unit == 'JY/PIXEL':
                gsmcube[ichan, :, :] = self.i_nu(frequency)
            elif unit == 'K':
                gsmcube[ichan, :, :] = self.t_b(frequency)

        gsmhdu = fits.PrimaryHDU(gsmcube)

        self.hdr3d.set('BUNIT', format(unit, '8'))

        gsmhdu.header = self.hdr3d

        if fitsfile.exists():
            wrng_msg = f"{fitsfile} already exists. Overwriting"
            errh.issue_warning(UserWarning, wrng_msg)

        gsmhdu.writeto(fitsfile, overwrite=True)

    def generate_miriad_image(self, miriad_image: pathlib.Path,
                              unit: str = 'JY/PIXEL'):
        """
        Write .fits file cube of intensities or brightness temperatures

        Parameters
        ----------
        miriad_image
            Full path to write miriad image to
        unit
            One of either 'JY/PIXEL' (default) or 'K' for intensity or
            brightness temperature, respectively
        Returns
        -------
        None
        """
        temp_dir = miriad_image.resolve().parent
        temp_pfx = temp_dir.joinpath(str(miriad_image.name).replace('.', '_'))
        temp_fits_file = pathlib.Path(f"{temp_pfx}_"
                                      f"temp_{_generate_random_chars(10)}.fits")

        if temp_fits_file.exists():
            raise FileExistsError("I don't know how this is possible. You "
                                  "literally had a 1 in 800 quadrillion chance "
                                  "of randomly generating the same name as a "
                                  "file that already exists")

        self.generate_fits(temp_fits_file, unit=unit)

        if miriad_image.exists():
            errh.issue_warning(UserWarning,
                               f"{str(miriad_image)} already exists, removing")
            shutil.rmtree(miriad_image)

        miriad.fits(_in=temp_fits_file, out=miriad_image, op='xyin')
        temp_fits_file.unlink()

    @abstractmethod
    def t_b(self, freq: float) -> np.ndarray:
        """
        Calculate SkyModel brightness temperature [K]

        Parameters
        ----------
        freq
           Frequency at which brightness temperatures are wanted [Hz]

        Returns
        -------
        np.ndarray of brightness temperatures with shape (self.n_x, self.n_y)
        """

    @abstractmethod
    def i_nu(self, freq: float) -> np.ndarray:
        """
        Calculate SkyModel intensity image [Jy / pixel]

        Parameters
        ----------
        freq
           Frequency at which intensities are wanted [Hz]

        Returns
        -------
        np.ndarray of intensities with shape (self.n_x, self.n_y)
        """


class LoadedSkyModel(SkyModel):
    """Sky model loaded from a .fits file"""
    def __init__(self, npix: Tuple[int, int], cdelt: float, coord0: SkyCoord,
                 tb_data: np.ndarray):
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
        tb_data
            numpy.ndarray containing temperature brightness data
        """
        super().__init__(npix, cdelt, coord0)
        self.data = tb_data

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_data):
        self._data = new_data

    def t_b(self, freq: float) -> np.ndarray:
        freq_present = self.frequency_present(freq)
        if not freq_present:
            errh.raise_error(ValueError,
                             f"{freq}Hz not present in model frequencies")
        idx = self.frequencies.index(freq_present)

        return self._data[idx]

    def i_nu(self, freq: float) -> np.ndarray:
        wavelength = con.c / freq
        pix_sr = (self.cdelt / 180. * con.pi) ** 2.
        kelvin_to_jansky_per_pixel = 2. * con.Boltzmann * pix_sr / \
                                     (wavelength ** 2.) / 1e-26

        return self.t_b(freq) * kelvin_to_jansky_per_pixel


class DiffuseSkyModel(SkyModel):
    """Diffuse sky model for Galactic emission"""
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
            One of 'GSM2016' (default),
        """
        if model not in DiffuseSkyModel.VALID_MODELS:
            errh.raise_error(ValueError,
                             f"{model} not a valid diffuse model")

        super().__init__(npix, cdelt, coord0)
        self.model = model

    def t_b(self, freq: float) -> np.ndarray:
        if self.model == 'GSM2016':
            gdsm = GlobalSkyModel2016(freq_unit='Hz', data_unit='TCMB',
                                      resolution='hi')
        else:
            gdsm = None
            errh.raise_error(ValueError,
                             f"{self.model} is not a recognised model for "
                             f"DiffuseSkyModel class")

        temp_fitsfile = pathlib.Path('temp.fits')

        gdsm.generate(freq)
        gdsm.write_fits(temp_fitsfile.__str__())  # expects str type
        hdugsm = fits.open(temp_fitsfile)
        hdugsm[1].header.set('COORDSYS', 'G')

        temp_fitsfile.unlink()

        return np.single(reproject_from_healpix(hdugsm[1], self.hdr2d))[0]

    def i_nu(self, freq: float) -> np.ndarray:
        wavelength = con.c / freq
        pix_sr = (self.cdelt / 180. * con.pi) ** 2.
        kelvin_to_jansky_per_pixel = 2. * con.Boltzmann * pix_sr / \
                                     (wavelength ** 2.) / 1e-26

        return self.t_b(freq) * kelvin_to_jansky_per_pixel


class SmallSkyModel(SkyModel):
    """Small-scale sky model for Galactic emission"""
    VALID_MODELS = ('MHD',)  # Valid Galactic sky models

    def __init__(self, npix: Tuple[int, int], cdelt: float, coord0: SkyCoord,
                 model: str = 'MHD'):
        """
        Extends the behaviour of superclass constructor method by requiring a
        model for the Galactic small_scale emission

        Parameters
        ----------
        npix
            Number if pixels in (x, y) of the SkyModel
        cdelt
            Pixel size [deg]
        coord0
            Central coordinate of SkyModel
        model
            One of 'MHD' (default),
        """
        if model not in DiffuseSkyModel.VALID_MODELS:
            errh.raise_error(ValueError,
                             f"{model} not a valid short-scale model")

        super().__init__(npix, cdelt, coord0)
        self.model = model

        if self.model == 'MHD':
            hduim = fits.open(DATA_FILES[self.model])
            model_hdr = hduim[0].header
            freqs = np.linspace(
                model_hdr['CRVAL3'] - ((model_hdr['CRPIX3'] - 1) *
                                       model_hdr['CDELT3']),
                model_hdr['CRVAL3'] + ((model_hdr['NAXIS3'] -
                                        model_hdr['CRPIX3']) *
                                       model_hdr['CDELT3']),
                model_hdr['NAXIS3']
            )
            self.add_frequency(freqs)

    def t_b(self, freq: float) -> np.ndarray:
        if self.model == 'MHD':
            hduim = fits.open(DATA_FILES[self.model])
            imdat = np.squeeze(hduim[0].data)
            imhdr = hduim[0].header

    def i_nu(self, freq: float) -> np.ndarray:
        wavelength = con.c / freq
        pix_sr = (self.cdelt / 180. * con.pi) ** 2.
        kelvin_to_jansky_per_pixel = 2. * con.Boltzmann * pix_sr / \
                                     (wavelength ** 2.) / 1e-26

        return self.t_b(freq) * kelvin_to_jansky_per_pixel


if __name__ == '__main__':
    import logging
    import astropy.units as u

    coord = SkyCoord(1.34, 0.4312, frame='fk5', unit=(u.rad, u.rad))
    dsm = DiffuseSkyModel((102, 48), 1./12., coord)
    dsm.add_frequency(np.linspace(3.9e9, 4.1e9, 3))
    test_fitsfile = pathlib.Path("~/Desktop/test.fits")

    test_fitsfile = test_fitsfile.expanduser()
    dsm.generate_fits(test_fitsfile)
