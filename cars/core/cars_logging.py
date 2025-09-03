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
import logging.config
import os
import platform

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


PROGRESS = 21
logging.addLevelName(PROGRESS, "PROGRESS")
PROFILING_LOG = 15
logging.addLevelName(PROFILING_LOG, "PROFILING_LOG")

profiling_logger = logging.getLogger("profiling_logger")


class ProfilingFilter(logging.Filter):  # pylint: disable=R0903
    """
    ProfilingFilter
    """

    def filter(self, record):
        """ "
        Filter message
        """
        return "PROFILING_LOG" not in record.msg


class BasicFilter(logging.Filter):  # pylint: disable=R0903
    """
    ProfilingFilter
    """

    def filter(self, record):
        """ "
        Filter message
        """
        return "PROFILING_LOG" in record.msg


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


def setup_logging(
    loglevel="PROGRESS",
    out_dir=None,
    log_dir=None,
    pipeline="",
    in_worker=False,
    global_log_file=None,
):
    """
    Setup the CARS logging configuration

    :param loglevel: log level default WARNING
    """
    # logging
    if loglevel == "PROGRESS":
        numeric_level = PROGRESS
    else:
        if isinstance(loglevel, int):
            numeric_level = loglevel
        else:
            numeric_level = getattr(logging, loglevel, None)

    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % loglevel)

    def add_handler_name(config, handler_name, filtered_logger=None):
        """
        add handler name in known handlers of loggers
        """
        for key in config["loggers"].keys():
            if filtered_logger is not None:
                if key in filtered_logger:
                    config["loggers"][key]["handlers"].append(handler_name)
            else:
                config["loggers"][key]["handlers"].append(handler_name)

    logging_config = {
        "version": 1,
        "disable_existing_loggers": True,
        "formatters": {
            "standard": {
                "format": "%(asctime)s :: %(levelname)s ::  %(message)s"
            },
            "workers": {
                "format": (
                    "%(asctime)s :: %(levelname)s "
                    + ":: %(thread)d :: %(process)d :: %(message)s"
                )
            },
        },
        "handlers": {
            "stdout": {
                "level": numeric_level,
                "formatter": "standard",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "filters": ["no_profiling"],
            }
        },
        "filters": {
            "no_profiling": {"()": ProfilingFilter},
            "only_profiling": {"()": BasicFilter},
        },
        "loggers": {
            "": {  # root logger
                "handlers": [],
                "level": min(numeric_level, PROFILING_LOG),
                "propagate": False,
            },
            "cars": {
                "handlers": [],
                "level": min(numeric_level, PROFILING_LOG),
                "propagate": False,
            },
            "__main__": {  # if __name__ == '__main__'
                "handlers": [],
                "level": min(numeric_level, PROFILING_LOG),
                "propagate": False,
            },
        },
    }

    # add global log file
    if global_log_file is not None:
        os.makedirs(os.path.dirname(global_log_file), exist_ok=True)

        handler_global_main = "file_global_main"
        logging_config["handlers"][handler_global_main] = {
            "class": "logging.FileHandler",
            "filename": global_log_file,
            "level": min(numeric_level, logging.INFO),
            "mode": "w",
            "formatter": "standard",
            "filters": ["no_profiling"],
        }
        add_handler_name(logging_config, handler_global_main)

    # add file formaters:
    if out_dir is not None:
        log_file = os.path.join(
            out_dir,
            "{}_{}.log".format(
                datetime.now().strftime("%y-%m-%d_%Hh%Mm"), pipeline
            ),
        )
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handler_main = "file_main"
        logging_config["handlers"][handler_main] = {
            "class": "logging.FileHandler",
            "filename": log_file,
            "level": min(numeric_level, logging.INFO),
            "mode": "w",
            "formatter": "standard",
            "filters": ["no_profiling"],
        }
        add_handler_name(logging_config, handler_main)

        # profiling for main
        profiling_dir = os.path.join(out_dir, "profiling")
        if not os.path.exists(profiling_dir):
            os.makedirs(profiling_dir)
        profiling_file = os.path.join(profiling_dir, "profiling.log")

        handler_main_profiling = "file_main_profiling"
        logging_config["handlers"][handler_main_profiling] = {
            "class": "logging.FileHandler",
            "filename": profiling_file,
            "level": PROFILING_LOG,
            "mode": "w",
            "formatter": "standard",
            # "filters": ["only_profiling"],
        }
        add_handler_name(logging_config, handler_main_profiling)

    if not in_worker:
        add_handler_name(logging_config, "stdout")
    else:
        # remove stdout as handler
        del logging_config["handlers"]["stdout"]

        # add file handlers

        # change level of root logger in workerss
        handler_workers = "file_workers"
        handler_workers_profiling = "file_workers_profiling"
        logging_config["loggers"]["profiling_logger"] = {
            "handlers": [],
            "level": PROFILING_LOG,
            "propagate": False,
        }
        # logging.getLogger().setLevel(min(numeric_level, PROFILING_LOG))
        # profiling_logger.setLevel(min(numeric_level, PROFILING_LOG))

        # sett handlers
        log_file_workers = os.path.join(
            log_dir,
            "workers.log",
        )
        os.makedirs(os.path.dirname(log_file_workers), exist_ok=True)

        logging_config["handlers"][handler_workers] = {
            "class": "cars.core.cars_logging.WorkerHandler",
            "filename": log_file_workers,
            "level": min(numeric_level, logging.INFO),
            "formatter": "workers",
            "filters": ["no_profiling"],
        }
        add_handler_name(logging_config, handler_workers)

        # profiling
        log_file_workers_profiling = os.path.join(
            log_dir,
            "profiling.log",
        )

        logging_config["handlers"][handler_workers_profiling] = {
            "class": "cars.core.cars_logging.ProfilinglHandler",
            "filename": log_file_workers_profiling,
            "level": PROFILING_LOG,
            "formatter": "workers",
            # "filters": ["only_profiling"],
        }
        add_handler_name(logging_config, handler_workers_profiling)

    # Create config
    logging.config.dictConfig(logging_config)


def add_progress_message(message):
    """
    Add enforced message with INFO level
    to stdout and logging file

    :param message: logging message
    """
    logging.log(PROGRESS, message)


def add_profiling_message(message):
    """
    Add enforced message with PROFILING_LOG level
    to stdout and logging file

    :param message: logging message
    """
    logging.log(PROFILING_LOG, message)


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
            setup_logging(loglevel=log_level, log_dir=log_dir, in_worker=True)
            res = func(*args, **kwargs)
        except Exception as worker_error:
            logging.exception(worker_error, exc_info=True)
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
        setup_logging(loglevel=log_level, log_dir=log_dir, in_worker=True)
        res = func(*args, **kwargs)
    except Exception as worker_error:
        logging.exception(worker_error, exc_info=True)
        raise worker_error
    return res
