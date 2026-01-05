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
Contains tools for multiprocessing
"""

import logging
import multiprocessing
import os
import re
import subprocess

from cars.orchestrator.orchestrator import get_available_ram, get_total_ram


def replace_data(list_or_dict, func_to_apply, *func_args):
    """
    Replace MpJob in list or dict by their real data
    (can deal with FactorizedObject)

    :param list_or_dict: list or dict of data or mp_objects.FactorizedObject
    :param func_to_apply: function to apply
    :param func_args: function arguments

    :return: list or dict with real data
    :rtype: list, tuple, dict, mp_objects.FactorizedObject
    """
    if (
        isinstance(list_or_dict, (list, tuple))
        and len(list_or_dict) == 1
        and type(list_or_dict[0]).__name__ == "FactorizedObject"
    ):
        # list_or_dict is a single FactorizedObject
        factorized_object = list_or_dict[0]
        args = factorized_object.get_args()
        args = replace_data_rec(args, func_to_apply, *func_args)
        kwargs = factorized_object.get_kwargs()
        kwargs = replace_data_rec(kwargs, func_to_apply, *func_args)

        factorized_object.set_args(args)
        factorized_object.set_kwargs(kwargs)
        return [factorized_object]

    return replace_data_rec(list_or_dict, func_to_apply, *func_args)


def replace_data_rec(list_or_dict, func_to_apply, *func_args):
    """
    Replace MpJob in list or dict by their real data recursively

    :param list_or_dict: list or dict of data
    :param func_to_apply: function to apply
    :param func_args: function arguments

    :return: list or dict with real data
    :rtype: list, tuple, dict
    """

    if isinstance(list_or_dict, (list, tuple)):
        res = []
        for arg in list_or_dict:
            if isinstance(arg, (list, tuple, dict)):
                res.append(replace_data_rec(arg, func_to_apply, *func_args))
            else:
                res.append(func_to_apply(arg, *func_args))
        if isinstance(list_or_dict, tuple):
            res = tuple(res)

    elif isinstance(list_or_dict, dict):
        res = {}
        for key, value in list_or_dict.items():
            if isinstance(value, (list, dict, tuple)):
                res[key] = replace_data_rec(value, func_to_apply, *func_args)
            else:
                res[key] = func_to_apply(value, *func_args)

    else:
        raise TypeError(
            "Function only support list or dict or tuple, "
            "but type is {}".format(list_or_dict)
        )

    return res


def get_slurm_data():
    """
    Get slurm data
    """

    def get_data(chain, pattern):
        """
        Get data from pattern

        :param chain: chain of character to parse
        :param pattern: pattern to find

        :return: found data
        """

        match = re.search(pattern, chain)
        value = None
        if match:
            value = match.group(1)
        return int(value)

    on_slurm = False
    slurm_nb_cpu = None
    slurm_max_ram = None
    try:
        sub_res = subprocess.run(
            "scontrol show job $SLURM_JOB_ID",
            shell=True,
            capture_output=True,
            text=True,
            check=False,
        )
        slurm_infos = sub_res.stdout

        slurm_nb_cpu = get_data(slurm_infos, r"ReqTRES=cpu=(\d+)")
        slurm_max_ram = get_data(slurm_infos, r"ReqTRES=cpu=.*?mem=(\d+)")
        # convert to Mb
        slurm_max_ram *= 1024
        logging.info("Available CPUs  in SLURM : {}".format(slurm_nb_cpu))
        logging.info("Available RAM  in SLURM : {}".format(slurm_max_ram))

    except Exception as exc:
        logging.debug("Not on Slurm cluster")
        logging.debug(str(exc))
    if slurm_nb_cpu is not None and slurm_max_ram is not None:
        on_slurm = True

    return on_slurm, slurm_nb_cpu, slurm_max_ram


def compute_conf_auto_mode(is_windows, max_ram_per_worker):
    """
    Compute confuration to use in auto mode

    :param is_windows: True if runs on windows
    :type is_windows: bool
    :param max_ram_per_worker: max ram per worker in MB
    :type max_ram_per_worker: int
    """

    on_slurm, nb_cpu_slurm, max_ram_slurm = get_slurm_data()

    if on_slurm:
        available_cpu = nb_cpu_slurm
    else:
        available_cpu = (
            multiprocessing.cpu_count()
            if is_windows
            else len(os.sched_getaffinity(0))
        )
        logging.info("available cpu : {}".format(available_cpu))

    if available_cpu == 1:
        logging.warning("Only one CPU detected.")
        available_cpu = 2
    elif available_cpu == 0:
        logging.warning("No CPU detected.")
        available_cpu = 2

    if on_slurm:
        ram_to_use = max_ram_slurm
    else:
        ram_to_use = get_total_ram()
        logging.info("total ram :  {}".format(ram_to_use))

    # use 50% of total ram
    ram_to_use *= 0.5

    possible_workers = int(ram_to_use // max_ram_per_worker)
    if possible_workers == 0:
        logging.warning("Not enough memory available : failure might occur")
    nb_workers_to_use = max(1, min(possible_workers, available_cpu - 1))

    logging.info("Number of workers : {}".format(nb_workers_to_use))
    logging.info("Max memory per worker : {} MB".format(max_ram_per_worker))

    # Check with available ram
    available_ram = get_available_ram()
    if int(nb_workers_to_use) * int(max_ram_per_worker) > available_ram:
        logging.warning(
            "CARS will use 50% of total RAM, "
            " more than currently available RAM"
        )

    return int(nb_workers_to_use)
