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

# Standard imports
import sys
from datetime import datetime

PROGRESS = 21
logging.addLevelName(PROGRESS, "PROGRESS")


def create(loglevel="PROGRESS"):
    """
    Setup the CARS logging configuration

    :param loglevel: log level default WARNING
    """
    # logging
    if loglevel == "PROGRESS":
        numeric_level = PROGRESS
    else:
        numeric_level = getattr(logging, loglevel, None)

    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % loglevel)

    logging.basicConfig(
        stream=sys.stdout,
        level=numeric_level,
        datefmt="%y-%m-%d %H:%M:%S",
        format="%(asctime)s :: %(levelname)s :: %(message)s",
    )


def add_progress_message(message):
    """
    Add enforced message with INFO level
    to stdout and logging file

    :param message: logging message
    """
    logging.log(PROGRESS, message)


def add_log_file(out_dir, command):
    """
    Add dated file handler to the logger.

    :param out_dir: output directory in which the log file will be created
    :type out_dir: str
    :param command: command name which will be part of the log file name
    :type command: str
    """
    # set file log handler
    now = datetime.now()
    h_log_file = logging.FileHandler(
        os.path.join(
            out_dir,
            "{}_{}.log".format(now.strftime("%y-%m-%d_%Hh%Mm"), command),
        )
    )

    formatter = logging.Formatter(
        fmt="%(asctime)s :: %(levelname)s :: %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
    )
    h_log_file.setFormatter(formatter)
    h_log_file.setLevel(logging.getLogger().getEffectiveLevel())
    logging.getLogger().addHandler(h_log_file)


def wrap_logger(func, log_dir, log_level):
    """
    Wrapper logger function to wrap worker func
    and setup the worker logger
    :param func: wrapped function
    :param log_dir: output directory of worker logs
    :param log_level: logging level of the worker logs
    """

    def wrapper_builder(*args, **kwargs):
        """
        Wrapper function

        :param argv: args of func
        :param kwargs: kwargs of func
        """
        # init logger
        try:
            setup_logger(log_dir, log_level)
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
        setup_logger(log_dir, log_level)
        res = func(*args, **kwargs)
    except Exception as worker_error:
        logging.exception(worker_error, exc_info=True)
        raise worker_error
    return res


def setup_logger(log_dir, log_level):
    """
    Setup the worker logger inside wrapper
    :param log_dir: output directory of worker logs
    :param log_level: logging level of the worker logs
    """
    cars_logger = logging.getLogger()
    if len(logging.getLogger().handlers) < 1:
        formatter = logging.Formatter(
            fmt="%(asctime)s :: %(levelname)s "
            + ":: %(thread)d :: %(process)d :: %(message)s",
            datefmt="%y-%m-%d %H:%M:%S",
        )
        if log_level == logging.DEBUG:
            formatter = logging.Formatter(
                fmt="%(asctime)s :: %(module)s "
                + ":: %(levelname)s :: %(thread)d "
                + ":: %(process)d :: %(message)s",
                datefmt="%y-%m-%d %H:%M:%S",
            )
        if os.path.exists(log_dir):
            h_log_file = logging.FileHandler(
                os.path.join(
                    log_dir,
                    "workers.log",
                )
            )
            h_log_file.setFormatter(formatter)
            h_log_file.setLevel(log_level)
            cars_logger.addHandler(h_log_file)
            cars_logger.setLevel(log_level)
