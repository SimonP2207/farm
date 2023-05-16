"""Classes for creation and handling of calibration tables and their errors"""
from typing import Union

import numpy as np
import numpy.typing as npt

import colorednoise as cn


def sefd_to_rms(sefd: Union[float, npt.NDArray],
                n_ant: int, t_total: float, bandwidth: float, n_pol: int = 2,
                efficiency: float = 1.) -> Union[float, npt.NDArray]:
    """
    System equivalent flux density (SEFD) to root-mean-square (rms) image noise
    [Jy]

    Parameters
    ----------
    sefd
        System equivalent flux density [Jy]
    n_ant
        Number of antennae in the array
    t_total
        Total on source time [s]
    bandwidth
        Bandwidth [Hz]
    n_pol
        Number of polarizations projects used in the image, which should be 2
        for Stokes I, Q, U or V, and 1 for LCP, RCP, XX or YY. Default is 2
    efficiency
        Correlator efficiency. Values must be in the range 0 to 1. The default
        is 1 (i.e. no losses)

    Returns
    -------
    Image noise level [Jy]
    """
    n_baselines = n_ant * (n_ant - 1.)
    sqrt = np.sqrt(n_pol * n_baselines * t_total * bandwidth)

    return sefd / (efficiency * sqrt)


# def generate_gain_errors(
#         n_int: int, n_freq: int, n_stations: int, rseed: int = 12345,
#         t_beta: float = 2., t_mean_amp: float = 1., t_mean_phase: float = 0.,
#         t_std_amp: float = 0.1, t_std_phase: float = 0.1,
#         f_beta: float = 2., f_mean_amp: float = 1., f_mean_phase: float = 0.,
#         f_std_amp: float = 0.1, f_std_phase: float = 0.1) -> npt.NDArray:
#     """
#     Generate array of time/frequency-dependent complex gain errors
#
#     Parameters
#     ----------
#     n_int
#         Number of time integration steps
#     n_freq
#         Number of frequency channels
#     n_stations
#         Number of stations in telescope array
#     rseed
#         Random number generation seed. Defaults to 12345
#     t_beta
#         Power-spectrum distribution exponent for time-dependent errors.
#         Defaults
#         to 2 (brown noise)
#     t_mean_amp
#         Average error in amplitude for time-dependent errors. Defaults to 1
#     t_mean_phase
#         Average error in phase for time-dependent errors [deg]. Defaults to 0
#     t_std_amp
#         Standard deviation of amplitude for time-dependent errors.
#         Defaults to 0.1
#     t_std_phase
#         Standard deviation of phase for time-dependent errors [deg].
#         Defaults to 0.1
#     f_beta
#         Power-spectrum distribution exponent for frequency-dependent errors.
#         Defaults to 2 (brown noise)
#     f_mean_amp
#         Average error in amplitude for frequency-dependent errors.
#         Defaults to 1
#     f_mean_phase
#         Average error in phase for frequency-dependent errors [deg].
#         Defaults to
#         0
#     f_std_amp
#         Standard deviation of amplitude for frequency-dependent errors.
#         Defaults
#         to 0.1
#     f_std_phase
#         Standard deviation of phase for frequency-dependent errors [deg].
#         Defaults to 0.1
#
#     Returns
#     -------
#     Array of complex gain errors of array-shape (num_int, num_freq, num_tel)
#     """
#     gains = np.empty((n_int, n_freq, n_stations), dtype=np.csingle)
#
#     # Convert degrees to radians
#     t_mean_phase_rad = np.radians(t_mean_phase)
#     f_mean_phase_rad = np.radians(f_mean_phase)
#     t_std_phase_rad = np.radians(t_std_phase)
#     f_std_phase_rad = np.radians(f_std_phase)
#
#     def random_gains(beta: float, n: int, av: float, std: float,
#                      seed: int) -> npt.NDArray:
#         """Produce random noise values (applicable to amp or phase)"""
#         g0 = cn.powerlaw_psd_gaussian(beta, n, random_state=seed)
#         g = (g0 - np.average(g0)) * std / np.std(g0) + av
#
#         return g
#
#     for idx_station in range(n_stations):
#         # Time-dependent gains
#         t_amp = random_gains(t_beta, n_int, t_mean_amp, t_std_amp,
#                              rseed)
#         t_ph = random_gains(t_beta, n_int, t_mean_phase_rad, t_std_phase_rad,
#                             rseed + 1)
#         t_gains = t_amp * np.exp(1j * t_ph)
#
#         f_amp = random_gains(f_beta, n_freq, f_mean_amp, f_std_amp,
#                              rseed + 2)
#         f_ph = random_gains(f_beta, n_freq, f_mean_phase_rad, f_std_phase_rad,
#                             rseed + 3)
#         f_gains = f_amp * np.exp(1j * f_ph)
#
#         tf_gains = np.outer(t_gains, f_gains)
#         gains[:, :, idx_station] = tf_gains
#
#     return gains
#

class _BaseCalErrors:
    """
    Base class to handle calibration station-dependent error sub-classes for
    frequency-dependent bandpass and time-dependent gains
    """

    @staticmethod
    def _random_cal_errs(n: int, beta: float, av: float, std: float,
                         seed: int) -> npt.NDArray:
        """Produce random noise values (applicable to amp or phase)"""
        g0 = cn.powerlaw_psd_gaussian(beta, n, random_state=seed)
        # noinspection PyUnresolvedReferences
        g = (g0 - np.average(g0)) * std / np.std(g0) + av

        return g

    def __init__(self, n_steps: int, n_stations: int,
                 beta: float = 2., mean_amp: float = 1.,
                 mean_phase: float = 0., std_amp: float = 0.1,
                 std_phase: float = 1., rseed: int = 12345):
        self._rseed = rseed
        self.beta = beta
        self.mean_amp = mean_amp
        self.mean_phase = mean_phase
        self.std_amp = std_amp
        self.std_phase = std_phase

        self._errors = np.empty((n_stations, n_steps), dtype=np.csingle)
        self._errors_empty = True

    @property
    def _n(self) -> int:
        """Number of steps"""
        return np.shape(self._errors)[1]

    @property
    def rseed(self) -> int:
        """Random number generation seed"""
        return self._rseed

    @property
    def n_stations(self) -> int:
        """Number of stations/dishes related to the calibration tables"""
        return np.shape(self._errors)[0]

    @property
    def mean_phase_rads(self) -> float:
        """Mean phase of errors in radians"""
        return np.radians(self.mean_phase)

    @property
    def std_phase_rads(self) -> float:
        """Standard deviation of the phase errors in radians"""
        return np.radians(self.std_phase)

    @property
    def errors(self) -> npt.NDArray[np.csingle]:
        """
        Table of the complex calibration errors of shape (n_stations, n_steps)
        """
        if self._errors_empty:
            for idx_station in range(self.n_stations):
                amps = self._random_cal_errs(
                    self._n, self.beta, self.mean_amp, self.std_amp,
                    self.rseed
                )
                phases = self._random_cal_errs(
                    self._n, self.beta, self.mean_phase_rads,
                    self.std_phase_rads, self.rseed + 1
                )
                self._errors[idx_station] = amps * np.exp(1j * phases)
            self._errors_empty = False

        return self._errors


class GainsErrors(_BaseCalErrors):
    """Class for handling station/time dependent gains errors"""
    def __init__(self, n_times: int, n_stations: int,
                 beta: float = 2., mean_amp: float = 1.,
                 mean_phase: float = 0., std_amp: float = 0.1,
                 std_phase: float = 0.1, rseed: int = 12345):
        """
        Parameters
        ----------
        n_times
            Number of time intervals
        n_stations
            Number of telescope stations/dishes
        beta
            Exponent of generated noise whereby S(f) = (1 / f) ** beta. Defaults
            to 2 (brown noise)
        mean_amp
            Mean amplitude of errors. Defaults to 1
        mean_phase
            Mean phase of errors [deg]. Defaults to 0
        std_amp
            Standard deviation of error amplitudes. Defaults to 0.1
        std_phase
            Standard deviation of error phases [deg]. Defaults to 0.1
        rseed
            Random number generator seed. Defaults to 12345
        """
        super().__init__(n_times, n_stations,
                         beta, mean_amp, mean_phase, std_amp, std_phase,
                         rseed)

    @property
    def n_times(self):
        """Number of time step integrations"""
        return self._n


class BandpassErrors(_BaseCalErrors):
    """Class for handling station/frequency dependent bandpass errors"""
    def __init__(self, n_freq: int, n_stations: int,
                 beta: float = 2., mean_amp: float = 1.,
                 mean_phase: float = 0., std_amp: float = 0.1,
                 std_phase: float = 0.1, rseed: int = 12345):
        """
        Parameters
        ----------
        n_freq
            Number of frequency channels
        n_stations
            Number of telescope stations/dishes
        beta
            Exponent of generated noise whereby S(f) = (1 / f) ** beta. Defaults
            to 2 (brown noise)
        mean_amp
            Mean amplitude of errors. Defaults to 1
        mean_phase
            Mean phase of errors [deg]. Defaults to 0
        std_amp
            Standard deviation of error amplitudes. Defaults to 0.1
        std_phase
            Standard deviation of error phases [deg]. Defaults to 0.1
        rseed
            Random number generator seed. Defaults to 12345
        """
        super().__init__(n_freq, n_stations,
                         beta, mean_amp, mean_phase, std_amp, std_phase,
                         rseed)

    @property
    def n_freq(self):
        """Number of frequency channels"""
        return self._n


class CalibrationErrors:
    """
    Class for handling calibration errors consisting of station-dependent
    bandpass (frequency-dependent) and gains (time-dependent) errors. Note that
    these errors require that the gains errors and bandpass errors have the
    same number of stations/dishes. Also, errors are contiguous in time and
    frequency and therefore only applicable to a single Scan and Subband.
    """
    def __init__(self, gains_errs: GainsErrors, bpass_errs: BandpassErrors):
        """
        Parameters
        ----------
        gains_errs
            GainsErrors instance representing time/station-dependent gains
            calibration errors
        bpass_errs
            BandpassErrors instance representing time/frequency-dependent
            bandpass calibration errors
        """
        if bpass_errs.n_stations != gains_errs.n_stations:
            raise ValueError(
                f"Can't create calibration errors as number of stations for "
                f"bandpass errors ({bpass_errs.n_stations}) and for gains "
                f"errors ({gains_errs.n_stations}) are not equal"
            )

        self._gains_errs = gains_errs
        self._bpass_errs = bpass_errs

        self._errors_empty = True
        self._errors = np.empty(
            (gains_errs.n_times, bpass_errs.n_freq, bpass_errs.n_stations),
            dtype=np.csingle
        )

    @property
    def n_times(self) -> int:
        """
        Number of time step integrations on time-dependent gains errors
        """
        return np.shape(self._errors)[0]

    @property
    def n_freq(self) -> int:
        """
        Number of frequency channels in frequency-dependent bandpass errors
        """
        return np.shape(self._errors)[1]

    @property
    def n_stations(self) -> int:
        """
        Number of stations/dishes related to the calibration tables
        """
        return np.shape(self._errors)[2]

    @property
    def gains_errs(self) -> GainsErrors:
        """
        Associated gains errors as a GainsErrors instance
        """
        return self._gains_errs

    @gains_errs.setter
    def gains_errs(self, new_gains_errs: GainsErrors):
        if not isinstance(new_gains_errs, GainsErrors):
            raise TypeError(
                f"new_gains_errs is of type {type(new_gains_errs)} "
                f"not GainErrors"
            )

        if self.bpass_errs.n_stations != self.gains_errs.n_stations:
            raise ValueError(
                f"Can't replace gains errors as number of stations for "
                f"bandpass errors ({self.bpass_errs.n_stations}) and for new "
                f"gains errors ({self.gains_errs.n_stations}) are not equal"
            )

        self._gains_errs = new_gains_errs
        self._errors_empty = True

    @property
    def bpass_errs(self) -> BandpassErrors:
        """Associated bandpass errors as a BandpassErrors instance"""
        return self._bpass_errs

    @bpass_errs.setter
    def bpass_errs(self, new_bpass_errs: BandpassErrors):
        if not isinstance(new_bpass_errs, BandpassErrors):
            raise TypeError(
                f"new_gains_errs is of type {type(new_bpass_errs)} "
                f"not BandpassErrors"
            )

        if self.bpass_errs.n_stations != self.gains_errs.n_stations:
            raise ValueError(
                f"Can't replace gains errors as number of stations for "
                f"bandpass errors ({self.bpass_errs.n_stations}) and for new "
                f"gains errors ({self.gains_errs.n_stations}) are not equal"
            )

        self._bpass_errs = new_bpass_errs
        self._errors_empty = True

    @property
    def errors(self) -> npt.NDArray[np.csingle]:
        """Complex calibration errors of shape (n_times, n_freq, n_stations)"""
        if self._errors_empty:
            for i in range(self.n_stations):
                self._errors[:, :, i] = np.outer(self.gains_errs.errors[i],
                                                 self.bpass_errs.errors[i])
            self._errors_empty = False
        return self._errors


if __name__ == '__main__':
    num_stations = 512
    gains_e = GainsErrors(200, num_stations)
    bpass_e = BandpassErrors(100, num_stations)
    cal_errors = CalibrationErrors(gains_e, bpass_e)
    print(cal_errors.errors.shape)
