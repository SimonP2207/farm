try:
    import casatools as tools
    import casatasks as tasks
    raise ImportWarning("casatools/casatasks are not installed")
except ModuleNotFoundError:
    import os
    import subprocess
    import pathlib
    from typing import Union, Optional

    from farm import LOGGER
    from farm.software.common import which

    _CASA_PATH = which('casa')

    if _CASA_PATH is None:
        raise EnvironmentError("CASA is not in your path")

    class _Namespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def add_member(self, **kwargs):
            self.__dict__.update(kwargs)


    class _CasaTask:
        def __init__(self, name: str):
            from pathlib import Path
            from logging import FileHandler

            self.name = name

            # Find FARM log file if it exists
            self.logfile = None
            handlers = LOGGER.handlers
            is_file_handler = [isinstance(_, FileHandler) for _ in handlers]

            if any(is_file_handler):
                file_handler = handlers[is_file_handler.index(True)]
                self.logfile = Path(file_handler.baseFilename)

        @property
        def _command(self):
            return f"{self.name}(" +\
                   ', '.join([f"{k}={v.__repr__()}"
                              for k, v in self.kwargs.items()]) +\
                   ')'

        def __call__(self, **kwargs):

            self.kwargs = kwargs

            cmd = (f"{_CASA_PATH} --nogui --norc --agg --notelemetry "
                   f"--nocrashreport --nologger --log2term")

            if self.logfile:
                cmd += f" --logfile {self.logfile} "

            cmd += f' -c "{self._command}"'
            subprocess.run(cmd, shell=True)


    class CasaScript:
        def __init__(self, filename: Union[pathlib.Path, str],
                     logfile: Optional[Union[pathlib.Path, str]] = None):

            if isinstance(filename, str):
                filename = pathlib.Path(filename)
            self.file = filename
            self.file.unlink(missing_ok=True)
            self.file.touch()

            if isinstance(logfile, str):
                logfile = pathlib.Path(logfile)

            self.logfile = logfile

        def add_task(self, line):
            with open(self.file, 'at') as f:
                f.write(f"\n{line}")

        def execute(self):



            cmd = (f"{_CASA_PATH} --nogui --norc --agg --notelemetry "
                   f"--nocrashreport --nologger --log2term "
                   f"--logfile {self.logfile} -c {self.file}")
            subprocess.run(cmd, shell=True)

    tools = _Namespace()
    tasks = _Namespace()

    tasks.add_member(tclean=_CasaTask('tclean'),
                     exportuvfits=_CasaTask('exportuvfits'),
                     importuvfits=_CasaTask('importuvfits'),
                     vishead=_CasaTask('vishead'),
                     concat=_CasaTask('concat'),)
