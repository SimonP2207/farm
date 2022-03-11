"""
Methods/classes related to the command-line interface with miriad. Code is
copied from mirpy v0.3.1 by means of combining all contents of mirpy's
wrapper.py and commands.py into this single module.
"""
import os
import re
import subprocess
import sys
import warnings
import farm.errorhandling as errh
import farm.software.common as sfuncs

# __all__ = ["mir_commands", "MiriadError", "miriad"]
__all__ = ["miriad"]

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


@errh.log_errors_warnings
def mir_func(f, thefilter):
    """Wrapper around miriad system calls"""

    def func(*args, **kw):
        if len(args) == 1:
            kw["_in"] = args[0]
        args = to_args(kw)

        # SJDP added: 14/02/2022, for no-limit character-length input parameters
        # In case any of the args have values > 80 characters long. However,
        # this doesn't work even though the miriad user guide says using @file
        # containing long parameter values is ok.
        # TODO: Contact Mark Wieringa or someone at ATNF to ask why @file
        #  containing filenames > 80 characters doesn't work
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
            warnings.warn(msg)

        if proc.returncode != 0:
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
    print(sfuncs.which('miriad'))
    a = miriad.fits(_in='test.fits', op='xyin', out='test.mirim')
