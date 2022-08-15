"""
Methods/classes related to the command-line interface with miriad. Code is
copied from mirpy v0.3.1 by means of combining all contents of mirpy's
wrapper.py and commands.py into this single module, apart from the 'which'
method which is located in the farm.software.common module
"""
import os
import re
import shutil
import subprocess
import sys
import warnings
import pathlib
from typing import ByteString, Tuple
import numpy as np
from numpy import typing as npt

from .. import LOGGER
from . import common as sfuncs
from ..miscellaneous import error_handling as errh, decorators

MIR_CHAR_LIMIT = 64
LONG_KEYS = ('tin', 'out', 'vis', 'model')

if sfuncs.which('miriad') is None:
    errh.raise_error(ImportError, "miriad is not in your PATH")


def mir_commands():
    """Get a filter list of miriad commands in the miriad bin directory"""
    mir = sfuncs.which('miriad')
    if mir is None:
        raise OSError('miriad is not available. Check your PATH.')
    mirpath = os.path.split(mir)[0]
    return [cmd for cmd in os.listdir(mirpath)
            if not (cmd.startswith('mir') or
                    cmd.startswith('doc') or
                    cmd.startswith('pgxwin') or
                    cmd.startswith('pgdisp') or
                    cmd.endswith('.exe'))]


def _remove_path(path):
    if isinstance(path, str):
        path = pathlib.Path(str)

    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


@decorators.log_errors_warnings
def mir_func(f, thefilter):
    """Wrapper around miriad system calls"""
    from .. import LOGGER

    def func(*args, **kw):
        # Added to provide shortened parameter names so miriad's maximum
        # character limit is not exceeded
        original_args = to_args(kw)
        reformat_args = False

        # puthd indicates whether the vis/in parameter value takes extra
        # characters on the end of its path e.g. /my/data/image.im/crval1
        puthd = True if str(f) in ('puthd', 'gethd', 'delhd') else False
        for k, v in kw.items():
            if isinstance(v, pathlib.Path):
                v = str(v)
                kw[k] = v
            if k in LONG_KEYS or match_in(k):
                if ',' in v:
                    v = v.split(',')
                if not isinstance(v, list):
                    if len(str(v)) > MIR_CHAR_LIMIT:
                        reformat_args = True
                        break
                else:
                    for v_ in v:
                        if len(str(v_)) > MIR_CHAR_LIMIT:
                            reformat_args = True
                            break
        # If shortening of key-values is required, shorten to single letter
        # names, starting from 'A'
        if reformat_args:
            mv_dict = {}
            ord_num = ord('A')
            for k, v in kw.items():
                if len(str(v)) <= MIR_CHAR_LIMIT:
                    continue
                if isinstance(v, pathlib.Path):
                    v = str(v)

                if ',' in v:
                    v = v.split(',')

                if not isinstance(v, list):
                    if k in LONG_KEYS or match_in(k):
                        next_path_name = chr(ord_num)
                        _remove_path(pathlib.Path(next_path_name))

                        ord_num += 1
                        if puthd and match_in(k):
                            mv_dict[next_path_name] = os.sep.join(v.split(os.sep)[:-1])
                            next_path_name += os.sep + v.split(os.sep)[-1]
                            kw[k] = next_path_name
                        else:
                            mv_dict[next_path_name] = v
                            kw[k] = next_path_name
                else:
                    new_v = []
                    for v_ in v:
                        next_path_name = chr(ord_num)
                        _remove_path(pathlib.Path(next_path_name))

                        ord_num += 1
                        mv_dict[next_path_name] = v_
                        new_v.append(next_path_name)
                        kw[k] = new_v

            # In case of shortened input args, rename original to shortened
            for k, v in mv_dict.items():
                if os.path.exists(v):
                    os.rename(v, k)

        if len(args) == 1:
            kw["_in"] = args[0]
        args = to_args(kw)

        # Execute miriad task with arguments
        LOGGER.info(' '.join([f] + original_args))
        proc = subprocess.Popen([f] + args, shell=False, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate()

        # SJDP added: 10/02/22
        try:
            stdout = stdout.decode(sys.stdout.encoding)
        except UnicodeDecodeError:
            stdout = stdout.decode('latin1')

        try:
            stderr = stderr.decode(sys.stdout.encoding)
        except UnicodeDecodeError:
            stderr = stderr.decode('latin1')

        lines = stderr.split('\n')
        warns = []
        errors = []
        for line in lines:
            wpfx = "### Warning: "
            epfx = "### Fatal Error: "
            if line.startswith(wpfx):
                warns.append(line[len(wpfx):])
            elif line.startswith(epfx):
                errors.append(line[len(epfx) + 1:])
            else:
                errors.append(line)
        if warns:
            msg = "'%s': " % f
            msg += "\n".join(warns)
            LOGGER.warning(msg)

        # Rename shortened file/directory names back to originals
        if reformat_args:
            # In case of shortened input args, rename back to original
            for k, v in mv_dict.items():
                try:
                    os.rename(k, v)
                except FileNotFoundError as err:
                    # Raise exception that renamed file/directory not present
                    # ONLY if miriad ran without errors. In that case, miriad
                    # wouldn't produce output files/directories and renaming
                    # will fail with error. Therefore, skip the error and rename
                    # any input files to ensure the MiriadError below is raised
                    if proc.returncode == 0:
                        raise err

        if proc.returncode != 0:
            LOGGER.error("\n".join(errors))
            raise MiriadError("\n".join(errors))

        out = stdout.strip()
        if thefilter is not None:
            return thefilter(out)

        return out

    return func


def to_args(kw):
    """Turn a key dictionary into a list of k=v command-line arguments."""
    out = []
    for k, v in kw.items():
        if match_in(k):
            k = "in"
        if isinstance(v, list) or isinstance(v, tuple):
            v = ",".join([str(i) for i in v])
        out.append("%s=%s" % (k, v))
    return out


def match_in(key):
    """Is the key  the 'in' keyword"""
    rx = re.compile("^_?([iI][nN])_?$")
    match = rx.match(key)
    if match:
        return True
    return False


class MiriadError(Exception):
    """An exception class for errors in miriad calls"""

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)


class Miriad(object):
    """
    A wrapper for miriad commands.
    Miriad commands are turned into python functions, e.g.

        fits in=example.uv out=example.fits op=uvout

    would be

        miriad.fits(_in="example.uv", out="blah.fits", op="uvout")


    NOTE!!! All miriad keys are turned into the same key name in python
    EXCEPT 'in' which is can't be used in python. The python function accepts
    any of 'In', 'IN', '_in', or 'in_' instead.
    """

    def __init__(self):
        self._common = mir_commands()
        self._filters = {}

    def __dir__(self):
        return self._common + ['set_filter']

    def set_filter(self, funcname, ffunc):
        """Set a filter function to filter stdout for a miriad command.
        Example:

             def uselessfilter(output):
                 return output.split('\\n')
             miriad.set_filter('uvindex', uselessfilter)

        Executing miriad.uvindex will nor return a list of strings (the lines
        in the stdout output.
        This can of course be used for useful filtering.
        """
        assert funcname in self._common
        self._filters[funcname] = ffunc

    def __getattr__(self, k):
        if k in self._common:
            thefilter = self._filters.get(k, None)
            fn = mir_func(k, thefilter)
            fn.__doc__ = self._help(k)
            fn.__name__ = k
            return fn
        else:
            return object.__getattribute__(self, k)

    @staticmethod
    def _help(taskname):
        p = subprocess.Popen('miriad', shell=True, stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # SJDP changed 'utf-8' to sys.stdin.encoding: 10/02/22
        p.stdin.write("{} {}\n{}\n".format("help", taskname, "exit")
                      .encode(sys.stdin.encoding))

        # SJDP added: 10/02/22
        try:
            stdout = p.communicate()[0].decode(sys.stdout.encoding)
        except UnicodeDecodeError:
            stdout = p.communicate()[0].decode('latin1')

        return stdout

    @staticmethod
    def is_miriad_vis_file(filename: pathlib.Path) -> bool:
        """
        Determine if a file is a miriad visibility data file or not

        Parameters
        ----------
        filename : pathlib.Path
            Path to prospective miriad visibility file

        Returns
        -------
        bool
            Whether filename is a miriad visibility file (True)
            or not (False)
        """
        if not filename.is_dir():
            return False

        req_contents = ['flags', 'header', 'history', 'vartable', 'visdata']
        contents = list(filename.iterdir())

        return all([filename / f in contents for f in req_contents])

    @staticmethod
    def write_mir_gains_table(mirfile: pathlib.Path, gheader: ByteString,
                              gtimes: npt.NDArray, ggains: npt.NDArray):
        """
        Write a set of gains to the gains table of a miriad visibility
        data file

        Parameters
        ----------
        mirfile : pathlib.Path
            Full path to miriad visibility data file
        gheader : ByteString
            Gains header
        gtimes : npt.NDArray
            Array of times corresponding to each gains interval
        ggains : npt.NDArray
            Array of gains of shape (number of antennae, 2)

        Returns
        -------
        None
        """
        from struct import pack

        n = len(gtimes)
        ngains = ggains.shape[1]
        with open(mirfile / 'gains', 'wb') as f:
            f.write(gheader)
            for i in range(n):
                f.write(pack('>d', gtimes[i]))
                f.write(pack(f'>{ngains * 2:.0f}f',
                             *ggains[i, :, :].flatten().tolist()))

    @classmethod
    def read_mir_gains_table(cls, mirfile: pathlib.Path) -> Tuple[
        ByteString, npt.NDArray, npt.NDArray]:
        """
        Read the gains table from a miriad visibility data file and return
        its header, times and gains

        Parameters
        ----------
        mirfile : pathlib.Path
            Full path to miriad visibility data file

        Returns
        -------
        Tuple[ByteString, npt.NDArray, npt.NDArray]
            Tuple of gains header, times and gain values. The latter is of
            shape (number of antennae, 2)

        Raises
        ------
        ValueError
            If mirfile is not a miriad visibility data file, or it is but
            does not contain a gains table
        """
        from struct import unpack

        if not cls.is_miriad_vis_file(mirfile):
            err_msg = f'{mirfile} is not miriad visibility data'
            errh.raise_error(ValueError, err_msg)

        if not (mirfile / 'gains').exists():
            err_msg = f'{mirfile} does not contain a gains table'
            errh.raise_error(ValueError, err_msg)

        # read header items we need
        ngains, nfeed, ntau, nsols = 0, 0, 0, 0
        items = [b'ngains', b'nfeeds', b'ntau', b'nsols']

        # Parse necessary information from the header
        with open(mirfile / 'header', 'rb') as f:
            line = f.read(16)
            while line:
                ln = unpack('!16B', line)[15]
                # round up to multiple of 16
                ln = 16 * ((ln + 15) // 16)
                item = unpack('!15s', line[0:15])[0].split(b'\0')[0]
                data = f.read(ln)

                if item in items:
                    val = unpack('!i', data[4:8])[0]
                    if item == b'nfeeds':
                        nfeeds = val
                    if item == b'ngains':
                        ngains = val
                    if item == b'ntau':
                        ntau = val
                    if item == b'nsols':
                        nsols = val
                line = f.read(16)

        n = max(1, nsols)
        with open(mirfile / 'gains', 'rb') as f:
            gtimes = np.zeros(n)
            ggains = np.zeros((n, ngains, 2), dtype=np.float32)
            gheader = f.read(8)
            for i in range(n):
                buf = f.read(8)
                time = unpack('>d', buf)
                gtimes[i] = time[0]
                buf = f.read(ngains * 8)
                g = unpack(f'>{ngains * 2:.0f}f', buf)
                ggains[i, :, :] = np.array(g).reshape(ngains, 2)

        return gheader, gtimes, ggains

    @classmethod
    def implement_gain_errors(cls, vis_file: pathlib.Path, t_interval: float,
                              pnoise: float, gnoise: float, rseed: int):
        """
        Introduce gains errors in to a miriad visibility data file

        Parameters
        ----------
        vis_file : pathlib.Path
            Full path to miriad visibility data file
        t_interval : float
            Interval between gain solutions [minutes]
        pnoise : float
            Phase error [deg]
        gnoise : float
            Amplitude error [percentage]
        rseed : int
            Random number generator seed
        """
        from numpy.random import default_rng

        if t_interval <= 60.:
            err_msg = "t_interval must be longer than 1h or get BP problem"
            errh.raise_error(ValueError, err_msg)

        if not cls.is_miriad_vis_file(vis_file):
            err_msg = f'{vis_file} is not miriad visibility data'
            errh.raise_error(ValueError, err_msg)

        if not (vis_file / 'gains').exists():
            # First run gperror to make a gain table with some nominal values
            # (since random number seed can not be passed)
            LOGGER.info(f"Creating gains table in {vis_file}")
            miriad.gperror(vis=vis_file, interval=t_interval,
                           pnoise=pnoise, gnoise=gnoise)

        # Now read the gain table and replace with some nice random numbers
        gheader, gtimes, ggains = cls.read_mir_gains_table(vis_file)

        phas_rms = pnoise * np.pi / 180.
        gain_rms = gnoise / 100.
        rng = default_rng(seed=rseed)
        gvals = rng.normal(loc=1., scale=gain_rms,
                           size=ggains[:, :, 0].shape)
        pvals = rng.normal(loc=0., scale=phas_rms,
                           size=ggains[:, :, 1].shape)
        cvals = (np.cos(pvals) + 1j * np.sin(pvals)) * gvals
        rvals = cvals.real.astype('float32')
        ivals = cvals.imag.astype('float32')
        my_ggains = np.stack((rvals, ivals), axis=2)
        cls.write_mir_gains_table(vis_file, gheader, gtimes, my_ggains)

    @staticmethod
    def write_mir_bandpass_table(mirfile: pathlib.Path, bheader: ByteString,
                                 btimes: npt.NDArray, bgains: npt.NDArray):
        """
        Write a set of gains to the bandpass table of a miriad visibility
        data file

        Parameters
        ----------
        mirfile : pathlib.Path
            Full path to miriad visibility data file
        bheader : ByteString
            Gains header
        btimes : npt.NDArray
            Array of times corresponding to bandpass solution interval
        bgains : npt.NDArray
            Array of bandpass solutions of shape (number of antennae,
            number of antennae, number of channels, 2)

        Returns
        -------
        None
        """
        from struct import pack

        n = len(btimes)
        nbpsols = n
        if n == 1 and btimes[0] == 0:
            nbpsols = 0

        ngains, nchan = bgains.shape[1:3]
        with open(mirfile / 'bandpass', 'wb') as f:
            f.write(bheader)
            for i in range(n):
                f.write(pack(f'>{ngains * nchan * 2:.0f}f',
                             *bgains[i, :, :, :].flatten().tolist()))
                if nbpsols > 0:
                    f.write(pack('>d', btimes[i]))

    @staticmethod
    def write_mir_freqs_table(mirfile: pathlib.Path, fheader: ByteString,
                              nchan: int, freq0: float, chan_width: float):
        from struct import pack

        with open(mirfile / 'freqs', 'wb') as f:
            f.write(fheader)
            f.write(pack('>iidd', nchan, 0, freq0 / 1e9, chan_width / 1e9))

    @classmethod
    def implement_bandpass_errors(cls, vis_file: pathlib.Path, nchan: int,
                                  freq0: float, chan_width: float,
                                  pnoise: float, gnoise: float, rseed: int):
        """
        Introduce bandpass errors in to a miriad visibility data file

        Parameters
        ----------
        vis_file : pathlib.Path
            Full path to miriad visibility data file
        nchan : int
            Number of channels in data file
        pnoise : float
            Phase error [deg]
        gnoise : float
            Amplitude error [percentage]
        rseed : int
            Random number generator seed
        """
        from numpy.random import default_rng

        rng = default_rng(seed=rseed)

        gheader, gtimes, ggains = cls.read_mir_gains_table(vis_file)

        bgains = np.zeros((ggains.shape[0], ggains.shape[1], nchan, 2),
                          dtype='float32')
        gvals = rng.normal(loc=1., scale=gnoise, size=bgains[:, :, :, 0].shape)
        pvals = rng.normal(loc=0., scale=pnoise, size=bgains[:, :, :, 1].shape)
        cvals: npt.NDArray = (np.cos(pvals) + 1j * np.sin(pvals)) * gvals
        rvals = cvals.real.astype('float32')
        ivals = cvals.imag.astype('float32')

        my_bgains = np.stack((rvals, ivals), axis=3)
        Miriad.write_mir_bandpass_table(vis_file, gheader, gtimes, my_bgains)
        cls.write_mir_freqs_table(vis_file, gheader, nchan, freq0, chan_width)

        miriad.puthd(_in=f'{str(vis_file)}/nbpsols', value=len(gtimes))
        miriad.puthd(_in=f'{str(vis_file)}/nspect0', value=1.)
        miriad.puthd(_in=f'{str(vis_file)}/nchan0', value=nchan)


miriad = Miriad()

# Uncomment if you want all tasks explicitly in exported namespace. Just takes
# a while to load them all...
# for task in mir_commands():
#     locals()[task] = miriad.__getattr__(task)
# __all__ += mir_commands()

if __name__ == '__main__':
    a = miriad.fits(_in='test.fits', op='xyin', out='test.im')
