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
Contains abstract function for abstract dask Cluster
"""

# Standard imports
import logging
import os
import time

# Third party imports
from abc import abstractmethod

import dask
import numpy as np
import psutil
import xarray as xr
import yaml
from dask.config import global_config as global_dask_config
from dask.config import set as dask_config_set
from dask.delayed import Delayed
from dask.distributed import as_completed
from dask.sizeof import sizeof as dask_sizeof
from distributed.diagnostics.plugin import WorkerPlugin
from distributed.utils import CancelledError

from cars.core import cars_logging

# CARS imports
from cars.orchestrator.cluster import abstract_cluster


class AbstractDaskCluster(abstract_cluster.AbstractCluster):
    """
    AbstractDaskCluster
    """

    def __init__(self, conf_cluster, out_dir, launch_worker=True):
        """
        Init function of AbstractDaskCluster

        :param conf_cluster: configuration for cluster

        """

        # call parent init
        super().__init__(conf_cluster, out_dir, launch_worker=launch_worker)
        # retrieve parameters
        self.nb_workers = self.checked_conf_cluster["nb_workers"]
        self.walltime = self.checked_conf_cluster["walltime"]
        self.use_memory_logger = self.checked_conf_cluster["use_memory_logger"]
        self.config_name = self.checked_conf_cluster["config_name"]
        self.profiling = self.checked_conf_cluster["profiling"]
        self.launch_worker = launch_worker

        self.activate_dashboard = self.checked_conf_cluster[
            "activate_dashboard"
        ]
        self.python = self.checked_conf_cluster["python"]
        if self.checked_conf_cluster["mode"] == "slurm_dask":
            self.account = self.checked_conf_cluster["account"]
            self.qos = self.checked_conf_cluster["qos"]

        if self.launch_worker:
            # Set DASK CARS specific config
            # TODO: update with adequate configuration through tests
            set_config()

            # Save dask config used
            save_config(self.out_dir, "dask_config_" + self.config_name)

            # Create cluster
            self.cluster, self.client = self.start_dask_cluster()

            # Add plugin to monitor memory of workers
            if self.use_memory_logger:
                plugin = ComputeDSMMemoryLogger(self.out_dir)
                self.client.register_worker_plugin(plugin)

    @abstractmethod
    def check_conf(self, conf):
        """
        Check configuration

        :param conf: configuration to check
        :type conf: dict

        :return: overloaded configuration
        :rtype: dict

        """

    @abstractmethod
    def start_dask_cluster(self):
        """
        Start dask cluster
        """

    def create_task_wrapped(self, func, nout=1):
        """
        Create task

        :param func: function
        :param nout: number of outputs
        """
        return dask.delayed(
            cars_logging.wrap_logger(func, self.worker_log_dir, self.log_level),
            nout=nout,
        )

    def get_delayed_type(self):
        """
        Get delayed type
        """
        return Delayed

    def start_tasks(self, task_list):
        """
        Start all tasks

        :param task_list: task list
        """

        return self.client.compute(task_list)

    def scatter(self, data, broadcast=True):
        """
        Distribute data through workers

        :param data: task data
        """
        return self.client.scatter(data, broadcast=broadcast)

    def future_iterator(self, future_list, timeout=None):
        """
        Start all tasks

        :param future_list: future_list list
        """

        return DaskFutureIterator(future_list, timeout=timeout)


class DaskFutureIterator:
    """
    iterator on dask futures, similar to as_completed
    Only returns the actual results, delete the future after usage
    """

    def __init__(self, future_list, timeout=None):  # pylint: disable=W0613
        # TODO: python 3.9: add timeout=timeout as parameter
        self.dask_a_c = as_completed(future_list, with_results=True)
        self.prev = None

    def __iter__(self):
        return self

    def __next__(self):
        try:
            fut, res = self.dask_a_c.__next__()
        except StopIteration as exception:
            if self.prev is not None:
                self.prev.cancel()
                self.prev = None
            raise exception
        except dask.distributed.TimeoutError as exception:
            raise TimeoutError("No tasks available") from exception
        # release previous future
        if self.prev is not None:
            self.prev.cancel()
            # store current future
            self.prev = fut

        if isinstance(res, CancelledError):
            raise RuntimeError("CancelError from worker {}".format(res))
        return res


def set_config():
    """
    Set particular DASK config such as:
    - scheduler
    """
    # TODO: export API to prepare.run and compute_dsm.run() to set scheduler
    # example mode debug: dask_config_set(scheduler='single-threaded')
    # example mode multithread: dask_config_set(scheduler='threads')
    # Here set Multiprocess mode instead multithread because of GIL blocking
    dask_config_set(scheduler="processes")


def save_config(output_dir: str, file_name: str):
    """
    Save DASK global config

    :param output_dir: output directory path
    :param file_name: output file name

    """
    logging.info(
        "Save DASK global merged config for debug "
        "(1: $DASK_DIR if exists, 2: ~/.config/dask/, ... ) "
    )
    # write global merged DASK config in YAML
    write_yaml_config(global_dask_config, output_dir, file_name)


def write_yaml_config(yaml_config: dict, output_dir: str, file_name: str):
    """
    Writes a YAML config to disk.
    TODO: put in global file if needed elsewhere than DASK conf save.

    :param yaml_config: YAML config to write
    :param output_dir: output directory path
    :param file_name: output file name
    """
    # file path where to store the dask config
    yaml_config_path = os.path.join(output_dir, file_name + ".yaml")
    with open(yaml_config_path, "w", encoding="utf-8") as yaml_config_file:
        yaml.dump(yaml_config, yaml_config_file)


@dask_sizeof.register_lazy("xarray")
def register_xarray():
    """
    Add hook to dask so it correctly estimates memory used by xarray
    """

    @dask_sizeof.register(xr.DataArray)
    # pylint: disable=unused-variable
    def sizeof_xarray_dataarray(xarr):
        """
        Inner function for total size of xarray_dataarray
        """
        total_size = dask_sizeof(xarr.values)
        for __, carray in xarr.coords.items():
            total_size += dask_sizeof(carray.values)
        total_size += dask_sizeof(xarr.attrs)
        return total_size

    @dask_sizeof.register(xr.Dataset)
    # pylint: disable=unused-variable
    def sizeof_xarray_dataset(xdat):
        """
        Inner function for total size of xarray_dataset
        """
        total_size = 0
        for __, varray in xdat.data_vars.items():
            total_size += dask_sizeof(varray.values)
        for __, carray in xdat.coords.items():
            total_size += dask_sizeof(carray)
        total_size += dask_sizeof(xdat.attrs)
        return total_size


class ComputeDSMMemoryLogger(WorkerPlugin):
    """A subclass of WorkerPlugin dedicated to monitoring workers memory

    This plugin enables two things:

    - Additional dask log traces (for each worker internal state change):

        - amount of tasks
        - associated memory
    - A numpy data file with memory metrics and timing
    """

    def __init__(self, outdir):
        """
        Constructor
        :param outdir: output directory
        :type outdir: string
        """
        self.outdir = outdir

    def setup(self, worker):
        """
        Associate plugin with a worker
        :param worker: The worker to associate the plugin with
        """
        # Pylint Exception : Inherited attributes outside __init__
        # pylint: disable=attribute-defined-outside-init
        self.worker = worker
        self.name = worker.name
        # Measure plugin registration time
        self.start_time = time.time()
        # Data will hold the memory traces as numpy array
        self.data = [[0, 0, 0, 0]]

    def transition(self, key, start, finish, **kwargs):
        """
        Callback when worker changes internal state
        """
        # TODO Pylint Exception : Inherited attributes outside __init__
        # pylint: disable=attribute-defined-outside-init

        # Define cumulants
        total_in_memory = 0
        total_nbytes = 0

        # Measure elapsed time for the state change
        elapsed_time = time.time() - self.start_time

        # Walk the worker known memory
        for task_key in self.worker.state.tasks.keys():
            task_size = self.worker.state.tasks[task_key].get_nbytes()

            total_in_memory += task_size
            total_nbytes += 1

        # Use psutil to capture python process memory as well
        process = psutil.Process(os.getpid())
        process_memory = process.memory_info().rss

        # Update data records
        self.data = np.concatenate(
            (
                self.data,
                np.array(
                    [
                        [
                            elapsed_time,
                            total_in_memory,
                            total_nbytes,
                            process_memory,
                        ]
                    ]
                ),
            )
        )
        # Convert nbytes size for logger
        total_nbytes = float(total_nbytes) / 1000000
        process_memory = float(process_memory) / 1000000

        # Log memory state
        logging.info(
            "Memory report: data created = {} ({} Mb), "
            "python process memory = {} Mb".format(
                total_in_memory,
                total_nbytes,
                process_memory,
            )
        )

        # Save data records in npy file
        # TODO: Save only every x seconds ?
        file = os.path.join(
            self.outdir, "dask_log", "memory_" + repr(self.name) + ".npy"
        )
        np.save(file, self.data)
