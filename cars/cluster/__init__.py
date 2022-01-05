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
from cars.cluster import dask_mode

#  Cluster module constants

# Cluster modes possibilities
CLUSTER_MODES = ["local_dask", "pbs_dask", "mp"]


def start_cluster(
    mode: str,
    nb_workers: int = None,
    walltime: str = None,
    out_dir: str = None,
    config_name: str = "unknown",
) -> Tuple[Cluster, Client, bool]:
    """
    CARS start generic cluster function

    This function:
    - checks mode in args possibilities
    - saves config (with config_name)
    - calls sub functions depending on the mode.

    :param mode: Parallelization mode Must be "local_dask", "pbs_dask" or "mp"
    :param nb_workers: Number of dask workers to use for the sift matching step
    :param walltime: Walltime of the dask workers
    :param out_dir: Output directory
    :param config_name: Name set in config saved file for dask mode.
    :return: Dask cluster or Mpcluster(TODO), Dask client, use dask boolean
    """
    # Check mode in args possibilities
    if mode not in CLUSTER_MODES:
        raise NotImplementedError("{} mode is not implemented".format(mode))

    logging.info("Start CARS cluster in {} mode".format(mode))

    # Prepare outputs
    cluster = None
    client = None
    use_dask = None

    if "dask" in mode:
        # DASK mode
        use_dask = True

        # Set DASK CARS specific config TODO.
        dask_mode.set_config()

        # Save dask config used
        dask_mode.save_config(out_dir, "dask_config_" + config_name)

        # Choose different cluster start depending on mode
        # TODO: do it with class polymorphism in refacto
        if mode == "local_dask":
            cluster, client = dask_mode.start_local_cluster(nb_workers)
        elif mode == "pbs_dask":
            cluster, client = dask_mode.start_pbs_cluster(
                nb_workers, walltime, out_dir
            )

        # Add plugin to monitor memory of workers
        plugin = dask_mode.ComputeDSMMemoryLogger(out_dir)
        client.register_worker_plugin(plugin)

    else:
        # No DASK mode
        use_dask = False

        if mode == "mp":
            # TODO: start_mp_cluster() for mp mode in cluster/mp_mode.py
            pass

    return cluster, client, use_dask


def stop_cluster(mode: str, cluster: Cluster, client: Client):
    """
    Stop cluster
    :param mode: Parallelization mode Must be "local_dask", "pbs_dask" or "mp"
    :param cluster: cluster to stop (only DASK for now)
    :param client: client to stop (DASK only for now)
    """
    # Check mode in args possibilities
    if mode not in CLUSTER_MODES:
        raise NotImplementedError("{} mode is not implemented".format(mode))

    # Choose different cluster stop depending on mode
    # TODO: do it with class polymorphism in refacto
    if mode == "local_dask":
        dask_mode.stop_local_cluster(cluster, client)
    elif mode == "pbs_dask":
        dask_mode.stop_pbs_cluster(cluster, client)
    elif mode == "mp":
        pass
        # TODO: clean mp cluster ?

    logging.info("Stop CARS cluster in {} mode".format(mode))
