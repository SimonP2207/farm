from typing import Union

import numpy as np
import numpy.typing as npt


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


def tec_screen(): ...
