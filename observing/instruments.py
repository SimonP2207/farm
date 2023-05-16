"""
Contains all classes/methods related to physical instrumentation:
- Station
- Telescope
- Correlator
"""
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Tuple, Union

import astropy.units as u
import numpy as np
import numpy.typing as npt
from astropy.coordinates import EarthLocation


class Subband:
    """
    Class for handling information for a subband, the constituent class of a
    Correlator setup
    """
    __instances = {}

    @classmethod
    @property
    def existing_bands(cls):
        """Tuple of existing band instances"""
        return [cls.__instances[n]() for n in cls.existing_bands_names]

    @classmethod
    @property
    def existing_bands_names(cls):
        """Tuple of existing band names"""
        return tuple(sorted(cls.__instances.keys()))

    @classmethod
    def __remove_subband(cls, subband: Union[str, 'Subband']):
        """Removes a subband from the class' records of instances"""
        if hasattr(subband, 'name'):
            band_name = subband.name
        elif isinstance(subband, str):
            band_name = subband
        else:
            raise TypeError(f"subband must be a str or Subband instance, not"
                            f"{type(subband)}")

        if band_name in cls.existing_bands_names:
            cls.__instances.pop(band_name)

    def __init__(self, freq0: float, chan_width: float, nchan: int,
                 name: str = None):
        """
        Parameters
        ----------
        freq0
            Minimum frequency of lowest channel's coverage [Hz]
        chan_width
            Channel width [Hz]
        nchan
            Number of channels
        name
            Name of subband
        """
        self.__freqs = np.arange(nchan) * chan_width + freq0

        if name is None:
            name = 'A'
            while name in self.__class__.existing_bands_names:
                name = chr(ord(name) + 1)
        # elif name in self.__class__.existing_bands_names:
        #     # raise ValueError(f"Band '{name}' already instantiated")
        #     self.__class__.__instances[name] = self

        self._name = name
        self.__class__.__instances[self.name] = self

    @property
    def name(self) -> str:
        """Subband name"""
        return self._name

    @property
    def nchan(self) -> int:
        """Number of channels in subband"""
        return len(self.frequencies)

    @property
    def chan_width(self) -> float:
        """Channel width [Hz]"""
        return self.frequencies[1] - self.frequencies[0]

    @property
    def frequencies(self) -> npt.NDArray[np.floating]:
        """Array of left-hand edges of all frequency channels"""
        return self.__freqs

    @property
    def freq0(self) -> float:
        """Minimum frequency of lowest channel's coverage [Hz]"""
        return self.frequencies[0]

    @property
    def bandwidth(self) -> float:
        """Total bandwidth of subband [Hz]"""
        return np.ptp(self.frequencies) + self.chan_width

    def __str__(self):
        tmplt = u"Band {}: {} \u00D7 {:.2f}{} channels from {:.1f} - {:.1f}{}"

        c_width, chan_unit = self.chan_width, 'MHz'
        if 1e5 > self.chan_width > 1e2:
            chan_unit = 'kHz'
        c_width /= 1e6 if chan_unit == 'MHz' else 1e3

        freq0, freqn, freq_unit = self.freq0, self.freq0 + self.bandwidth, 'MHz'
        if freqn > 1e9:
            freq_unit = 'GHz'

        freq0 /= 1e6 if freq_unit == 'MHz' else 1e9
        freqn /= 1e6 if freq_unit == 'MHz' else 1e9

        return tmplt.format(
            self.name, self.nchan, c_width, chan_unit, freq0, freqn, freq_unit
        )


class Correlator:
    """Class for handling all information pertaining to a Correlator setup"""
    def __init__(self, t_int: float, subbands: Optional[List[Subband]] = None):
        """
        Parameters
        ----------
        t_int
            Visibility integration time [s]
        subbands
            List of Subband instances to add to this Correlator (optional).
            Default is None
        """
        self._subbands = []
        self._t_int = t_int

        if subbands:
            self.add_subband(subbands)

    @property
    def t_int(self) -> float:
        """Visibility integration time [s]"""
        return self._t_int

    @property
    def subbands(self) -> List[Subband]:
        """List of Subbands in this Correlator setup"""
        return self._subbands

    def add_subband(self, new_subband: Union[Iterable[Subband], Subband]):
        """
        Add a sub-band to this Correlator setup

        Parameters
        ----------
        new_subband
            Subband instance to add to Correlator. Can be list of Subband
            instances
        """
        if isinstance(new_subband, Iterable):
            for subband in new_subband:
                self.add_subband(subband)

        else:
            # if not isinstance(new_subband, Subband):
            if not new_subband.__class__.__name__ == Subband.__name__:
                raise TypeError(f"{new_subband} is a {type(new_subband)} "
                                f"instance, not a {Subband} instance")

            self._subbands.append(new_subband)
            self._subbands.sort(key=lambda x: x.freq0)

    @property
    def frequencies(self) -> npt.NDArray[npt.NDArray[np.floating]]:
        """List of 1D numpy arrays of frequencies from all subbands"""
        return np.array([subband.frequencies for subband in self.subbands])

    @property
    def n_subbands(self) -> int:
        """Number of subbands in Correlator setup"""
        return len(self.subbands)

    def __str__(self):
        s = f"Correlator (t_int={self.t_int:.1f}s) with {self.n_subbands} " \
            f"subband(s):\n - "
        s += '\n - '.join([str(subband) for subband in self.subbands])

        return s


@dataclass
class Station:
    """
    Dataclass for handling individual station information of an interferometric
    array
    """
    station_model: Path
    position: Tuple[float, float, float]
    ants: Dict = field(init=False)
    n_ant: int = field(init=False)

    def __post_init__(self):
        ants_position_file = self.station_model / 'layout.txt'
        xs, ys = np.loadtxt(ants_position_file).T

        self.ants = {}
        for number in range(len(xs)):
            self.ants[number] = xs[number], ys[number]

        self.n_ant = len(self.ants)


@dataclass
class Telescope:
    """Class containing all information related to an interferometer"""
    model: Path
    ref_ant: Optional[int] = field(default=None)
    lon: float = field(init=False)
    lat: float = field(init=False)
    stations: Dict[int, Station] = field(init=False)
    n_stations: int = field(init=False)
    location: EarthLocation = field(init=False)
    centre: npt.NDArray[float] = field(init=False)
    _baseline_lengths: Union[None, Dict[Tuple[int, int], float]] = field(init=0)

    def summary(self) -> str:
        """Return summary of telescope as a string"""
        s = f"Telescope of {self.n_stations} stations located at " \
            f"lon={self.lon:.3f}deg, lat={self.lat:+.3f}deg whose reference " \
            f"antenna={self.ref_ant}"

        return s

    def __post_init__(self):
        self.lon, self.lat = np.loadtxt(self.model / 'position.txt')

        station_position_file = self.model / 'layout.txt'
        xs, ys, zs = np.loadtxt(station_position_file).T

        stations = []
        for f in self.model.iterdir():
            if f.is_dir() and 'station' in f.name:
                stations.append(f)
        stations.sort()

        self.stations = {}
        for s in stations:
            number = int(s.name.strip('station'))
            station_position = np.array([xs[number], ys[number], zs[number]])
            self.stations[number] = Station(stations[number], station_position)

        self.n_stations = len(self.stations)
        self.location = EarthLocation(lat=self.lat * u.deg,
                                      lon=self.lon * u.deg)
        self.centre = np.mean([v.position for k, v in self.stations.items()],
                              axis=0)

        # If no reference antenna is given, assign the central-most antenna in
        # the array as the reference antenna
        if self.ref_ant is None:
            station_dists = {}
            for k, s in self.stations.items():
                station_dists[k] = self.dist_to_centre(s.position)

            min_dist = 1e30
            for k, v in station_dists.items():
                if v < min_dist:
                    min_dist = v
                    self.ref_ant = k

        self._baseline_lengths = None

    @property
    def baseline_lengths(self) -> Dict[Tuple[int, int], float]:
        """
        Baseline lengths of all as a dict whose keys are 2-tuples of each
        baseline's antennae numbers and values are separations between those
        two antennae
        """
        if self._baseline_lengths is None:
            from itertools import combinations

            ant_pairs = combinations(self.stations.keys(), 2)
            self._baseline_lengths = {}
            for ant_pair in ant_pairs:
                positions = [self.stations[_].position for _ in ant_pair]
                baseline_length = self._dist_between_points(*positions)
                self._baseline_lengths[ant_pair] = baseline_length

        return self._baseline_lengths

    @staticmethod
    def _dist_between_points(pos1: npt.NDArray[float],
                             pos2: npt.NDArray[float]) -> float:
        """Dist [m] between two points"""
        return np.sqrt(np.sum((pos2 - pos1) ** 2.))

    def dist_to_centre(self, pos: npt.NDArray[float]) -> float:
        """Dist [m] to the array's geometrical centre"""
        return self._dist_between_points(pos, self.centre)

    def dist_to_refant(self, pos: npt.NDArray[float]) -> float:
        """Distance of a position to the reference antenna"""
        ref_station = self.stations[self.ref_ant]
        return self._dist_between_points(pos, ref_station.position)


if __name__ == '__main__':
    sb1 = Subband(1.1e8, 1e5, 100)
    sb2 = Subband(sb1.freq0 + sb1.bandwidth, 1e5, 150)
    sb3 = Subband(sb2.freq0 + sb2.bandwidth, 1e5, 200)
    print(Subband.existing_bands_names)
    sb4 = Subband(1e9, 1e6, 1e3, 'D')
    print(Subband.existing_bands_names)
    del sb4
    print(Subband.existing_bands_names)
    sb5 = Subband(1e9, 1e6, 1e3, 'D')
    print(Subband.existing_bands_names)
    correlator = Correlator(t_int=10.)
    correlator.add_subband((sb2, sb3, sb1))
    print(correlator)
