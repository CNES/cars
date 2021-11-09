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
CARS cluster module init file
"""

# Standard imports
import logging
from typing import Tuple

from distributed import Client

# Third party imports
from distributed.deploy import Cluster

# CARS imports
from cars.cluster.dask_mode import (
    ComputeDSMMemoryLogger,
    save_dask_config,
    set_dask_config,
    start_local_cluster,
    start_pbs_cluster,
    stop_local_cluster,
    stop_pbs_cluster,
)

# General variable of CARS cluster module

# Cluster modes possibilities
# True when DASK, False when not.
CLUSTER_MODES = {"local_dask": True, "pbs_dask": True, "mp": False}


def start_cluster(
    mode: str,
    nb_workers: int = None,
    walltime: str = None,
    out_dir: str = None,
    pipeline_name: str = "unknown",
) -> Tuple[Cluster, Client, bool]:
    """
    Cars start generic cluster function

    This function:
    - checks mode in args possibilities
    - save config
    - call sub functions depending on the mode.

    :param mode: Parallelization mode Must be "local_dask", "pbs_dask" or "mp"
    :param nb_workers: Number of dask workers to use for the sift matching step
    :param walltime: Walltime of the dask workers
    :param out_dir: Output directory
    :return: Dask cluster or Mpcluster, Dask client, use dask boolean
    """
    # Check mode in args possibilities
    if mode not in CLUSTER_MODES.keys():
        raise NotImplementedError("{} mode is not implemented".format(mode))

    cluster = None
    client = None

    if CLUSTER_MODES[mode]:
        # DASK mode

        # Set particular config TODO.
        set_dask_config()

        # Save dask config used
        save_dask_config(out_dir, "dask_config_" + pipeline_name)

        #
        if mode == "local_dask":
            cluster, client = start_local_cluster(nb_workers)
        elif mode == "pbs_dask":
            cluster, client = start_pbs_cluster(nb_workers, walltime, out_dir)

        # Add plugin to monitor memory of workers
        plugin = ComputeDSMMemoryLogger(out_dir)
        client.register_worker_plugin(plugin)

    else:
        if mode == "mp":
            # TODO: start_mp_cluster() for mp mode in cluster/mp_mode.py
            pass

    logging.info("Start CARS cluster in {} mode".format(mode))

    return cluster, client, CLUSTER_MODES[mode]


def stop_cluster(mode, cluster, client):
    """
    Stop cluster
    """
    # Check mode in args possibilities
    if mode not in CLUSTER_MODES.keys():
        raise NotImplementedError("{} mode is not implemented".format(mode))

    # TODO:

    if mode == "local_dask":
        stop_local_cluster(cluster, client)
    elif mode == "pbs_dask":
        stop_pbs_cluster(cluster, client)
    elif mode == "mp":
        pass
        # TODO: clean mp cluster ?

    logging.info("CARS cluster in {} mode stopped".format(mode))
