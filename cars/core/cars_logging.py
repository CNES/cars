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

import fcntl
import logging
import logging.config
import os

# Standard imports
from datetime import datetime
from functools import wraps

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


class ProfilinglHandler(logging.FileHandler):  # pylint: disable=R0903
    """
    Profiling
    """

    def __init__(self, log_file):
        """
        Init
        """
        self.sender = LogSender(log_file)
        logging.FileHandler.__init__(self, log_file, "a")

    def emit(self, record):
        """
        Emit
        """
        if "PROFILING" in record.levelname:
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
            fcntl.flock(file, fcntl.LOCK_EX)
            file.write(msg)
            fcntl.flock(file, fcntl.LOCK_UN)


def setup_logging(
    loglevel="PROGRESS",
    out_dir=None,
    log_dir=None,
    pipeline="",
    in_worker=False,
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

    def add_handler_name(config, handler_name):
        """
        add handler name in known handlers of loggers
        """
        for key in config["loggers"].keys():
            config["loggers"][key]["handlers"].append(handler_name)

    def add_handler_to_logging(
        log_file_name, formatter, log_level, handler_name
    ):
        """
        Add handler to logging
        """
        formatter_log = logging.Formatter(formatter)
        h_log_file = ProfilinglHandler(log_file_name)
        h_log_file.setFormatter(formatter_log)
        h_log_file.setLevel(log_level)
        h_log_file.set_name(handler_name)
        logging.getLogger().addHandler(h_log_file)

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
        "filters": {"no_profiling": {"()": ProfilingFilter}},
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

    # add file formaters:
    if out_dir is not None:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        log_file = os.path.join(
            out_dir,
            "{}_{}.log".format(
                datetime.now().strftime("%y-%m-%d_%Hh%Mm"), pipeline
            ),
        )
        handler_main = "file_main"
        logging_config["handlers"][handler_main] = {
            "class": "logging.FileHandler",
            "filename": log_file,
            "level": min(numeric_level, logging.INFO),
            "mode": "w",
            "formatter": "standard",
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
            "level": min(numeric_level, PROFILING_LOG),
            "mode": "w",
            "formatter": "standard",
        }
        add_handler_name(logging_config, handler_main_profiling)

    if not in_worker:
        add_handler_name(logging_config, "stdout")
        logging.config.dictConfig(logging_config)
    else:
        # remove stdout as handler
        del logging_config["handlers"]["stdout"]
        logging.config.dictConfig(logging_config)

        # add file handlers
        if log_dir is not None:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

        # change level of root logger in workerss
        logging.getLogger().setLevel(min(numeric_level, PROFILING_LOG))
        profiling_logger.setLevel(min(numeric_level, PROFILING_LOG))
        # Add filter to existing logger
        for handler in logging.root.handlers:
            handler.addFilter(ProfilingFilter())
        # sett handlers
        log_file_workers = os.path.join(
            log_dir,
            "workers.log",
        )
        handler_workers = "file_workers"
        add_handler_to_logging(
            log_file_workers,
            logging_config["formatters"]["workers"]["format"],
            min(numeric_level, logging.INFO),
            handler_workers,
        )

        # profiling
        log_file_workers_profiling = os.path.join(
            log_dir,
            "profiling.log",
        )
        handler_workers_profiling = "file_workers_profiling"
        add_handler_to_logging(
            log_file_workers_profiling,
            logging_config["formatters"]["workers"]["format"],
            min(numeric_level, PROFILING_LOG),
            handler_workers_profiling,
        )


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
