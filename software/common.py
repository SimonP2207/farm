"""
Common functions used between farm/software/*.py modules
"""
import os


def which(program: str):
    """Equivalent of unix 'which' command"""
    def is_exe(fpath_):
        """Is the file an executable?"""
        return os.path.exists(fpath_) and os.access(fpath_, os.X_OK)

    if program:
        fpath, fname = os.path.split(program)
    else:
        return None
    if fpath and is_exe(program):
        return program
    else:
        if "PATH" in os.environ:
            for path in os.environ["PATH"].split(os.pathsep):
                path = os.path.expandvars(os.path.expanduser(path))
                exe_file = os.path.join(path, program)
                if is_exe(exe_file):
                    return exe_file

    return None
