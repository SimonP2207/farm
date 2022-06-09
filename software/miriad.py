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


@decorators.log_errors_warnings
def mir_func(f, thefilter):
    """Wrapper around miriad system calls"""
    from .. import LOGGER

    def func(*args, **kw):
        # Added to provide shortened parameter names so miriad's maximum
        # character limit is not exceeded
        original_args = to_args(kw)
        reformat_args = False
        puthd = True if str(f) in ('puthd', 'gethd', 'delhd') else False
        for k, v in kw.items():
            if isinstance(v, pathlib.Path):
                v = str(v)
                kw[k] = v
            if k in LONG_KEYS or match_in(k):
                if not isinstance(v, list) and ',' not in v:
                    if len(str(v)) > MIR_CHAR_LIMIT:
                        reformat_args = True
                        break
                elif isinstance(v, list):
                    for v_ in v:
                        if len(str(v_)) > MIR_CHAR_LIMIT:
                            reformat_args = True
                            break
                elif ',' in v:
                    for v_ in v.split(','):
                        if len(str(v_)) > MIR_CHAR_LIMIT:
                            reformat_args = True
                            break
                else:
                    raise ValueError(f'help -> {k}={v}')

        if reformat_args:
            mv_dict = {}
            ord_num = 65
            for k, v in kw.items():
                if len(str(v)) <= MIR_CHAR_LIMIT:
                    continue
                if isinstance(v, pathlib.Path):
                    v = str(v)
                if not isinstance(v, list) and ',' not in v:
                    if k in LONG_KEYS or match_in(k):
                        next_path_name = chr(ord_num)

                        next_path = pathlib.Path(next_path_name)
                        if next_path.exists():
                            if next_path.is_dir():
                                shutil.rmtree(next_path)
                            else:
                                next_path.unlink()

                        ord_num += 1
                        if puthd and match_in(k):
                            mv_dict[next_path_name] = os.sep.join(v.split(os.sep)[:-1])
                            next_path_name += os.sep + v.split(os.sep)[-1]
                            kw[k] = next_path_name
                        else:
                            mv_dict[next_path_name] = v
                            kw[k] = next_path_name
                else:
                    if isinstance(v, list):
                        new_v = []
                        for v_ in v:
                            next_path_name = chr(ord_num)

                            next_path = pathlib.Path(next_path_name)
                            if next_path.exists():
                                if next_path.is_dir():
                                    shutil.rmtree(next_path)
                                else:
                                    next_path.unlink()

                            ord_num += 1
                            mv_dict[next_path_name] = v_
                            new_v.append(next_path_name)
                        kw[k] = new_v
                    elif ',' in v:
                        new_v = []
                        for v_ in v.split(','):
                            next_path_name = chr(ord_num)

                            next_path = pathlib.Path(next_path_name)
                            if next_path.exists():
                                if next_path.is_dir():
                                    shutil.rmtree(next_path)
                                else:
                                    next_path.unlink()

                            ord_num += 1
                            mv_dict[next_path_name] = v_
                            new_v.append(v_)
                        kw[k] = ','.join(new_v)
                    else:
                        raise ValueError(f'help -> {k}={v}')

            # In case of shortened input args, rename original to shortened
            for k, v in mv_dict.items():
                if os.path.exists(v):
                    os.rename(v, k)

        if len(args) == 1:
            kw["_in"] = args[0]
        args = to_args(kw)

        # SJDP added: 14/02/2022, for no-limit character-length input parameters
        # In case any of the args have values > 64 characters long. However,
        # this doesn't work even though the miriad user guide says using @file
        # containing long parameter values is ok.
        # TODO: Have to modify miriad's key.for file and recompile for increased
        #  buffer sizes
        # created_param_files = []
        # for idx, arg in enumerate(args):
        #     k, v = arg.split("=")
        #     if len(v) >= 0:
        #         param_file = pathlib.Path(f"{k}_param")
        #         if param_file.exists():
        #             param_file.unlink()
        #         with open(param_file, 'wt') as p:
        #             p.write(v)
        #         args[idx] = f"{k}=@{param_file}"
        #         created_param_files.append(param_file)
        #
        # temp_def_file = pathlib.Path(f"{f}.def")
        # if temp_def_file.exists():
        #     temp_def_file.unlink()
        #
        # with open(temp_def_file, 'wt') as def_file:
        #     def_file.write('\n'.join(args))
        #
        # proc = subprocess.Popen([f, '-f', temp_def_file], shell=False,
        #                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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

        if reformat_args:
            # In case of shortened input args, rename back to original
            for k, v in mv_dict.items():
                os.rename(k, v)

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


miriad = Miriad()
# Uncomment if you want all tasks explicitly in exported namespace. Just takes
# a while to load them all...
# for task in mir_commands():
#     locals()[task] = miriad.__getattr__(task)
# __all__ += mir_commands()

if __name__ == '__main__':
    a = miriad.fits(_in='test.fits', op='xyin', out='test.im')
