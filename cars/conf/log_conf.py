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
Logconf Cars module:
contains logging configuration
"""

# Standard imports
import logging
import os
from datetime import datetime


def setup_log(loglevel=logging.WARNING):
    """
    Setup the CARS logging configuration

    :param loglevel: log level default WARNING
    """
    # logging
    numeric_level = getattr(logging, loglevel, None)

    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % loglevel)

    logging.basicConfig(
        level=numeric_level,
        datefmt="%y-%m-%d %H:%M:%S",
        format="%(asctime)s :: %(levelname)s :: %(message)s",
    )


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
    h_log_file.setLevel(logging.getLogger().getEffectiveLevel())

    formatter = logging.Formatter(
        fmt="%(asctime)s :: %(levelname)s :: %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
    )
    h_log_file.setFormatter(formatter)

    # add it to the logger
    logging.getLogger().addHandler(h_log_file)
