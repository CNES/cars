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
Contains abstract function for Abstract Cluster
"""

# Standard imports
import logging
import os
from abc import ABCMeta, abstractmethod
from typing import Dict

# CARS imports
from cars.conf.input_parameters import ConfigType
from cars.core.utils import safe_makedirs
from cars.orchestrator.cluster import log_wrapper


class AbstractCluster(metaclass=ABCMeta):
    """
    AbstractCluster
    """

    # Available cluster modes to instanciate AbstractCluster subclasses.
    available_modes: Dict = {}

    # Define abstract attributes

    # profiling config parameter: activated, mode, loop_testing, memray
    profiling: ConfigType
    # cluster mode output directory
    out_dir: str

    def __new__(  # pylint: disable=W0613
        cls, conf_cluster, out_dir, launch_worker=True
    ):
        """
        Return the required cluster
        :raises:
         - KeyError when the required cluster is not registered

        :param conf_cluster: configuration for cluster
        :param out_dir: output directory for results
        :param launch_worker: launcher of the new worker
        :return: a cltser object
        """

        cluster_mode = "local_dask"
        if "mode" not in conf_cluster:
            logging.warning("Cluster mode not defined, default is used")
        else:
            cluster_mode = conf_cluster["mode"]

        if cluster_mode not in cls.available_modes:
            logging.error("No mode named {} registered".format(cluster_mode))
            raise KeyError("No mode named {} registered".format(cluster_mode))

        profiling = {"activated": False, "mode": "time", "loop_testing": False}
        if "profiling" in conf_cluster:
            conf_profiling = conf_cluster["profiling"]
            if "activated" in conf_profiling:
                profiling["activated"] = conf_profiling["activated"]
            if profiling["activated"]:
                if "mode" in conf_profiling:
                    profiling["mode"] = conf_profiling["mode"]
                if "loop_testing" in conf_profiling:
                    profiling["loop_testing"] = conf_profiling["loop_testing"]
                if "memray" in profiling["mode"]:
                    profiling_memory_dir = os.path.join(
                        out_dir, "profiling", "memray"
                    )
                    safe_makedirs(profiling_memory_dir, cleanup=True)
        conf_cluster["profiling"] = profiling
        logging.info("The AbstractCluster {} will be used".format(cluster_mode))
        cls.worker_log_dir = os.path.join(out_dir, "workers_log")
        if not os.path.exists(cls.worker_log_dir):
            os.makedirs(cls.worker_log_dir)
        cls.log_level = logging.getLogger().getEffectiveLevel()
        return super(AbstractCluster, cls).__new__(
            cls.available_modes[cluster_mode]
        )

    @classmethod
    def register_subclass(cls, short_name: str):
        """
        Allows to register the subclass with its short name
        :param short_name: the subclass to be registered
        :type short_name: string
        """

        def decorator(subclass):
            """
            Registers the subclass in the available methods
            :param subclass: the subclass to be registered
            :type subclass: object
            """
            cls.available_modes[short_name] = subclass
            return subclass

        return decorator

    def __init__(
        self, conf_cluster, out_dir, launch_worker=True
    ):  # pylint: disable=W0613
        """
        Init function of AbstractCluster

        :param conf_cluster: configuration for cluster

        """
        self.out_dir = out_dir
        # Check conf
        self.checked_conf_cluster = self.check_conf(conf_cluster)

    @abstractmethod
    def cleanup(self):
        """
        Cleanup cluster
        """

    @abstractmethod
    def check_conf(self, conf):
        """
        Check configuration

        :param conf: configuration to check
        :type conf: dict

        :return: overloaded configuration
        :rtype: dict

        """

    def get_conf(self):
        """
        Get overriden configuration


        :return: overriden configuration
        """

        return self.checked_conf_cluster

    def create_task(self, func, nout=1):
        """
        Create task

        :param func: function
        :param nout: number of outputs
        """

        def create_task_builder(*argv, **kwargs):
            """
            Create task builder to select the type of log
            according to the configured profiling mode

            :param argv: list of input arguments
            :param kwargs: list of named input arguments
            """
            if not self.profiling["activated"]:
                return self.create_task_wrapped(func, nout=nout)(
                    *argv, **kwargs
                )

            if self.profiling["mode"] == "time":
                (wrapper_func, additionnal_kwargs) = log_wrapper.LogWrapper(
                    func, self.profiling["loop_testing"]
                ).func_args_plus()
            elif self.profiling["mode"] == "cprofile":
                (
                    wrapper_func,
                    additionnal_kwargs,
                ) = log_wrapper.CProfileWrapper(func).func_args_plus()
            elif self.profiling["mode"] == "memray":
                (
                    wrapper_func,
                    additionnal_kwargs,
                ) = log_wrapper.MemrayWrapper(
                    func, self.profiling["loop_testing"], self.out_dir
                ).func_args_plus()
            return self.create_task_wrapped(wrapper_func, nout=nout)(
                *argv, **kwargs, **additionnal_kwargs
            )

        return create_task_builder

    @abstractmethod
    def create_task_wrapped(self, func, nout=1):
        """
        Create task

        :param func: function
        :param nout: number of outputs
        """

    @abstractmethod
    def start_tasks(self, task_list):
        """
        Start all tasks

        :param task_list: task list
        """

    @abstractmethod
    def scatter(self, data, broadcast=True):
        """
        Distribute data through workers

        :param data: task data
        """

    @abstractmethod
    def future_iterator(self, future_list):
        """
        Iterator, iterating on computed futures

        :param future_list: future_list list
        """
