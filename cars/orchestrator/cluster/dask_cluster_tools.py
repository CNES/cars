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
Contains functions cluster conf checker
"""
import time

from json_checker import And, Checker, Or


def create_checker_schema(conf):
    """
    Create checker shema that it can be overloaded

    :param conf: configuration to check
    :type conf: dict

    :return: overloaded configuration
    :rtype: dict

    """

    # init conf
    if conf is not None:
        overloaded_conf = conf.copy()
    else:
        conf = {}
        overloaded_conf = {}

    # Overload conf
    overloaded_conf["mode"] = conf.get("mode", "unknowed_dask")
    overloaded_conf["use_memory_logger"] = conf.get("use_memory_logger", False)
    overloaded_conf["nb_workers"] = conf.get("nb_workers", 2)
    overloaded_conf["max_ram_per_worker"] = conf.get("max_ram_per_worker", 2000)
    overloaded_conf["walltime"] = conf.get("walltime", "00:59:00")
    overloaded_conf["config_name"] = conf.get("config_name", "unknown")
    overloaded_conf["activate_dashboard"] = conf.get(
        "activate_dashboard", False
    )
    overloaded_conf["python"] = conf.get("python", None)
    overloaded_conf["profiling"] = conf.get("profiling", {})

    cluster_schema = {
        "mode": str,
        "use_memory_logger": bool,
        "nb_workers": And(int, lambda x: x > 0),
        "max_ram_per_worker": And(Or(float, int), lambda x: x > 0),
        "walltime": str,
        "config_name": str,
        "activate_dashboard": bool,
        "profiling": dict,
        "python": Or(None, str),
    }

    return overloaded_conf, cluster_schema


def check_configuration(overloaded_conf, cluster_schema):
    """
    Check configuration from overload conf and cluster schema


    :param conf: overloaded_conf to check
    :type conf: dict
    :param conf: cluster_schema checking rules
    :type conf: dict

    :return: overloaded configuration
    :rtype: dict

    """
    # Check conf
    checker = Checker(cluster_schema)
    checker.validate(overloaded_conf)

    # Check walltime format
    walltime = overloaded_conf["walltime"]
    try:
        time.strptime(walltime, "%H:%M:%S")
    except ValueError as err:
        raise ValueError("Walltime should be formatted as HH:MM:SS") from err

    return overloaded_conf
