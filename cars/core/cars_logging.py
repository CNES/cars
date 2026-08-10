#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2020 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of CARS
# (see https://github.com/CNES/cars).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
cCars logging module:
contains cars logging setup logger for main thread
and workers
"""

import logging
import os
import platform
import sys
import threading
from contextlib import contextmanager

# Standard imports
from datetime import datetime
from functools import wraps

SYS_PLATFORM = platform.system().lower()
IS_WIN = "windows" == SYS_PLATFORM

if IS_WIN:
    import msvcrt  # pylint: disable=E0401

    def lock(file):
        """Lock file for safe writing (Windows version)"""
        msvcrt.locking(file.fileno(), msvcrt.LK_LOCK, 0)

    def unlock(file):
        """Unlock file for safe writing (Windows version)"""
        msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 0)

else:
    import fcntl

    def lock(file):
        """Lock file for safe writing (Unix version)"""
        fcntl.flock(file, fcntl.LOCK_EX)

    def unlock(file):
        """Unlock file for safe writing (Unix version)"""
        fcntl.flock(file, fcntl.LOCK_UN)


PROFILING = 5  # we want DEBUG to not have profiling logs
logging.addLevelName(PROFILING, "PROFILING")


_WARNING_COUNTER = 0
_WARNING_LOCK = threading.Lock()  # for logs from workers


_LOGGER = None


class _LoggerProxy:  # pylint: disable=R0903
    """
    Proxy class to forward logging calls to the global logger instance.
    This allows us to use the logger as a module-level variable without
    having to pass it around explicitly.
    """

    def __getattr__(self, attr):
        return getattr(_get_logger(), attr)


logger = _LoggerProxy()


def _get_logger():
    """
    Helper function to get the global logger instance,
    creating it if necessary.
    """
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = logging.getLogger("CARS")
        _LOGGER.propagate = False
    return _LOGGER


@contextmanager
def mute_external_logging():
    """Temporarily mute global/root logging during external tool execution."""
    previous_disable = logging.root.manager.disable
    root_logger = logging.getLogger()
    previous_level = root_logger.level
    try:
        logging.disable(logging.CRITICAL)
        root_logger.setLevel(logging.CRITICAL + 1)
        yield
    finally:
        root_logger.setLevel(previous_level)
        logging.disable(previous_disable)


def reset_warning_count() -> None:
    """Reset global warning counting state for a fresh pipeline run."""
    global _WARNING_COUNTER
    with _WARNING_LOCK:
        _WARNING_COUNTER = 0


def get_warning_count() -> int:
    """Get total warning count captured by WarningCounterHandler."""
    with _WARNING_LOCK:
        return _WARNING_COUNTER


class WarningCounterHandler(logging.Handler):
    """In-process warning counter handler."""

    def emit(self, record):
        if record.levelno >= logging.WARNING:
            global _WARNING_COUNTER
            with _WARNING_LOCK:
                _WARNING_COUNTER += 1


class ProfilingFilter(logging.Filter):  # pylint: disable=R0903
    """
    ProfilingFilter - excludes profiling-level messages from standard logs
    """

    def filter(self, record):
        """
        Filter message - return False to exclude profiling messages
        """
        return record.levelno > PROFILING


class OnlyProfilingFilter(logging.Filter):  # pylint: disable=R0903
    """
    OnlyProfilingFilter - includes ONLY profiling-level messages
    """

    def filter(self, record):
        """
        Filter message - return True to include only profiling messages
        """
        return record.levelno == PROFILING


class SharelocFilter(logging.Filter):  # pylint: disable=R0903
    """
    SharelocFilter - filters out logs from shareloc module
    """

    def filter(self, record):
        """
        Filter message - return False to exclude shareloc logs
        """
        path_parts = record.pathname.split(os.sep)
        return "shareloc" not in path_parts


class ProfilinglHandler(logging.FileHandler):  # pylint: disable=R0903
    """
    Profiling
    """

    def __init__(self, filename, mode="a", encoding=None, delay=False):
        """
        Init
        """
        super().__init__(filename, mode, encoding, delay)
        self.sender = LogSender(filename)

    def emit(self, record):
        """
        Emit
        """
        if "PROFILING" in record.levelname:
            self.sender.write_log(self.format(record) + "\n")


class WorkerHandler(logging.FileHandler):  # pylint: disable=R0903
    """
    Profiling
    """

    def __init__(self, filename, mode="a", encoding=None, delay=False):
        """
        Init
        """
        super().__init__(filename, mode, encoding, delay)
        self.sender = LogSender(filename)

    def emit(self, record):
        """
        Emit
        """
        if "PROFILING" not in record.levelname:
            self.sender.write_log(self.format(record) + "\n")


class LogSender:  # pylint: disable=R0903
    """
    LogSender
    """

    def __init__(self, log_file):
        """
        Init
        """
        self.log_file = log_file

    def write_log(self, msg) -> None:
        """
        Write log
        """
        with open(self.log_file, "a", encoding="utf-8") as file:
            lock(file)
            file.write(msg)
            unlock(file)


def setup_logging_global(
    loglevel="INFO",
    global_log_file=None,
    use_stdout=True,
):
    """
    Setup global CARS logging configuration.
    Sets up the logger with stdout handler and main log file.
    This should be called only once in cars.py.

    :param loglevel: log level (default: "INFO")
    :param global_log_file: path to global log file (optional)
    :param use_stdout: whether to add stdout handler (default: True)
    """
    if isinstance(loglevel, int):
        numeric_level = loglevel
    else:
        numeric_level = getattr(logging, loglevel, None)

    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % loglevel)

    # Logger level is set to PROFILING so all messages reach handlers
    # each handler filters independently
    logger.setLevel(PROFILING)

    standard_formatter = logging.Formatter(
        "%(asctime)s :: %(levelname)s :: %(message)s"
    )

    # stdout handler: respects the user-selected level
    if use_stdout:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(numeric_level)
        stdout_handler.setFormatter(standard_formatter)
        stdout_handler.addFilter(ProfilingFilter())
        stdout_handler.addFilter(SharelocFilter())
        logger.addHandler(stdout_handler)

    # warning counter handler
    warningcounter_handler = WarningCounterHandler()
    warningcounter_handler.setLevel(logging.WARNING)
    logger.addHandler(warningcounter_handler)

    if global_log_file is not None:
        # global log file handler: at least DEBUG
        os.makedirs(os.path.dirname(global_log_file), exist_ok=True)
        global_log_file_handler = logging.FileHandler(global_log_file, mode="a")
        global_log_file_handler.setLevel(logging.DEBUG)
        global_log_file_handler.setFormatter(standard_formatter)
        global_log_file_handler.addFilter(ProfilingFilter())
        global_log_file_handler.addFilter(SharelocFilter())
        logger.addHandler(global_log_file_handler)


def setup_logging_pipeline(
    loglevel="INFO",
    out_dir=None,
    pipeline="",
):
    """
    Setup pipeline-specific logging configuration.
    Creates a log file for each pipeline step.
    Removes any previous pipeline-specific log handlers.

    :param loglevel: log level (default: "INFO")
    :param out_dir: output directory for pipeline logs
    :param pipeline: pipeline name (used in log filename)
    :return: path to the created log file
    """
    if isinstance(loglevel, int):
        numeric_level = loglevel
    else:
        numeric_level = getattr(logging, loglevel, None)

    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % loglevel)

    standard_formatter = logging.Formatter(
        "%(asctime)s :: %(levelname)s :: %(message)s"
    )

    if out_dir is None:
        return None

    # Create pipeline log file with timestamp
    log_file = os.path.join(
        out_dir,
        "{}_{}.log".format(
            datetime.now().strftime("%y-%m-%d_%Hh%Mm"), pipeline
        ),
    )
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    profiling_dir = os.path.join(out_dir, "profiling")
    os.makedirs(profiling_dir, exist_ok=True)

    # Remove handlers that were added for pipeline-specific logging
    # (keeping global and worker handlers)
    handlers_to_remove = []
    for handler in logger.handlers:
        # Skip custom worker handlers
        if isinstance(handler, (WorkerHandler, ProfilinglHandler)):
            continue
        if isinstance(handler, logging.FileHandler):
            handler_path = os.path.abspath(handler.baseFilename)
            out_dir_abs = os.path.abspath(out_dir)
            parent_logs_dir = os.path.dirname(out_dir_abs)

            # Remove handlers of previous pipeline runs
            # Keep:
            # - handlers directly in logs/ (global handler)
            # - handlers in workers_log/ (worker handlers)
            if (
                handler_path.startswith(parent_logs_dir)
                and "workers_log" not in handler_path
            ):
                # Check if handler is in a subdirectory
                rel_path = os.path.relpath(handler_path, parent_logs_dir)
                if os.sep in rel_path:  # handler is in a subdirectory
                    handlers_to_remove.append(handler)

    for handler in handlers_to_remove:
        logger.removeHandler(handler)
        handler.close()

    # out log file handler: at least DEBUG
    out_dir_log_file_handler = logging.FileHandler(log_file, mode="a")
    out_dir_log_file_handler.setLevel(logging.DEBUG)
    out_dir_log_file_handler.setFormatter(standard_formatter)
    out_dir_log_file_handler.addFilter(ProfilingFilter())
    out_dir_log_file_handler.addFilter(SharelocFilter())
    logger.addHandler(out_dir_log_file_handler)

    # profiling log file handler - only profiling level messages
    profiling_file = os.path.join(profiling_dir, "profiling.log")
    profiling_file_handler = logging.FileHandler(profiling_file, mode="a")
    profiling_file_handler.setLevel(PROFILING)
    profiling_file_handler.setFormatter(standard_formatter)
    profiling_file_handler.addFilter(OnlyProfilingFilter())
    logger.addHandler(profiling_file_handler)

    return log_file


def setup_logging_workers(
    loglevel="INFO",
    log_dir=None,
):
    """
    Setup worker-specific logging configuration :
    - profiling.log
    - workers.log

    :param loglevel: log level (default: "INFO")
    :param log_dir: directory for worker logs
    """
    if log_dir is None:
        return

    if isinstance(loglevel, int):
        numeric_level = loglevel
    else:
        numeric_level = getattr(logging, loglevel, None)

    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % loglevel)

    # Set logger level to PROFILING to allow all messages through to handlers
    logger.setLevel(PROFILING)

    workers_formatter = logging.Formatter(
        "%(asctime)s :: %(levelname)s :: %(thread)d :: "
        "%(process)d :: %(message)s"
    )

    # worker log file handler
    log_file_workers = os.path.join(
        log_dir,
        "workers.log",
    )
    os.makedirs(os.path.dirname(log_file_workers), exist_ok=True)

    worker_log_file_handler = WorkerHandler(log_file_workers, mode="a")
    worker_log_file_handler.setLevel(PROFILING)
    worker_log_file_handler.setFormatter(workers_formatter)
    logger.addHandler(worker_log_file_handler)

    # worker profiling log file handler
    log_file_workers_profiling = os.path.join(
        log_dir,
        "profiling.log",
    )
    worker_prof_log_file_handler = ProfilinglHandler(
        log_file_workers_profiling, mode="a"
    )
    worker_prof_log_file_handler.setLevel(PROFILING)
    worker_prof_log_file_handler.setFormatter(workers_formatter)
    logger.addHandler(worker_prof_log_file_handler)


def add_profiling_message(message):
    """
    Add enforced message with PROFILING level
    to stdout and logging file

    :param message: logging message
    """
    logger.log(PROFILING, message)


def wrap_logger(func, log_dir, log_level):
    """
    Wrapper logger function to wrap worker func
    and setup the worker logger
    :param func: wrapped function
    :param log_dir: output directory of worker logs
    :param log_level: logging level of the worker logs
    """

    @wraps(func)
    def wrapper_builder(*args, **kwargs):
        """
        Wrapper function

        :param argv: args of func
        :param kwargs: kwargs of func
        """
        # init logger
        try:
            setup_logging_workers(loglevel=log_level, log_dir=log_dir)
            res = func(*args, **kwargs)
        except Exception as worker_error:
            logger.exception(worker_error, exc_info=True)
            raise worker_error
        return res

    return wrapper_builder


def logger_func(*args, **kwargs):
    """
    Logger function to wrap worker func (with non local method)
    and setup the worker logger

    :param argv: args of func
    :param kwargs: kwargs of func
    """
    # Get function to wrap and id_list
    try:
        log_dir = kwargs["log_dir"]
        log_level = kwargs["log_level"]
        func = kwargs["log_fun"]
        kwargs.pop("log_dir")
        kwargs.pop("log_level")
        kwargs.pop("log_fun")
    except Exception as exc:  # pylint: disable=W0702 # noqa: B001, E722
        raise RuntimeError(
            "Failed in unwrapping. \n Args: {}, \n Kwargs: {}\n".format(
                args, kwargs
            )
        ) from exc
    # init logger
    try:
        setup_logging_workers(loglevel=log_level, log_dir=log_dir)
        res = func(*args, **kwargs)
    except Exception as worker_error:
        logger.exception(worker_error, exc_info=True)
        raise worker_error
    return res
