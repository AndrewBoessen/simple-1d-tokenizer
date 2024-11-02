import functools
import sys
import logging
from accelerate.logging import MultiProcessAdapter
from termcolor import colored
from iopath.common.file_io import PathManager as PathManagerClass

__all__ = ["setup_logger", "PathManager"]

PathManager = PathManagerClass()


class _ColorfulFormatter(logging.Formatter):
    """
    A custom logging formatter that adds color to warning and error messages.

    Parameters
    ----------
    *args : tuple
        Positional arguments passed to logging.Formatter.
    **kwargs : dict
        Keyword arguments passed to logging.Formatter.
        Must include 'root_name' and optionally 'abbrev_name'.

    Attributes
    ----------
    _root_name : str
        The base name for the logger, with a trailing dot.
    _abbrev_name : str
        Abbreviated logger name for shorter output, with a trailing dot.
    """

    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", self._root_name)
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        """
        Format the log message with optional color highlighting.

        Parameters
        ----------
        record : logging.LogRecord
            The log record to format.

        Returns
        -------
        str
            The formatted log message, with color if appropriate.
        """
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)

        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log

        return prefix + " " + log


@functools.lru_cache()
def setup_logger(
    name="TiTok",
    log_level: str = None,
    color=True,
    use_accelerate=True,
    output_file=None
):
    """
    Set up a logger with optional color output and file logging.

    Parameters
    ----------
    name : str, optional
        Name of the logger, by default "TiTok"
    log_level : str, optional
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        If None, defaults to DEBUG.
    color : bool, optional
        Whether to enable colored output, by default True
    use_accelerate : bool, optional
        Whether to wrap logger with MultiProcessAdapter, by default True
    output_file : str, optional
        Path to output log file. If None, file logging is disabled.

    Returns
    -------
    logging.Logger or MultiProcessAdapter
        Configured logger instance, wrapped in MultiProcessAdapter if use_accelerate=True

    Notes
    -----
    The logger outputs to both console and file (if specified) with the format:
    "[timestamp] logger_name level: message"

    Examples
    --------
    >>> logger = setup_logger(name="MyLogger", log_level="INFO")
    >>> logger.info("This is an info message")
    >>> logger.warning("This is a warning message")
    """
    logger = logging.getLogger(name)

    if log_level is None:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(log_level.upper())

    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%m/%d %H:%M:%S"
    )

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)

    if color:
        formatter = _ColorfulFormatter(
            colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
            datefmt="%m/%d %H:%M:%S",
            root_name=name,
        )
    else:
        formatter = plain_formatter

    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if output_file is not None:
        fileHandler = logging.FileHandler(output_file)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

    if use_accelerate:
        return MultiProcessAdapter(logger, {})
    else:
        return logger
