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

from cars.orchestrator.cluster.log_wrapper import cars_profile


class AbstractCluster(metaclass=ABCMeta):
    """
    AbstractCluster
    """

    # Available cluster modes to instanciate AbstractCluster subclasses.
    available_modes: Dict = {}

    # Define abstract attributes

    # cluster mode output directory
    out_dir: str

    def __new__(  # pylint: disable=too-many-positional-arguments
        cls,
        conf_cluster,
        out_dir,
        log_dir,
        launch_worker=True,
        data_to_propagate=None,
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

        cluster_mode = "multiprocessing"
        if conf_cluster is not None:
            if "mode" not in conf_cluster:
                logging.warning("Cluster mode not defined, default is used")
            else:
                cluster_mode = conf_cluster["mode"]

        if cluster_mode not in cls.available_modes:
            logging.error("No mode named {} registered".format(cluster_mode))
            raise KeyError("No mode named {} registered".format(cluster_mode))

        logging.info("The AbstractCluster {} will be used".format(cluster_mode))

        return super(AbstractCluster, cls).__new__(
            cls.available_modes[cluster_mode]
        )

    @classmethod
    def register_subclass(cls, *short_names: str):
        """
        Allows to register the subclass with its short name
        :param short_names: the subclasses to be registered
        :type short_names: string
        """

        def decorator(subclass):
            """
            Registers the subclass in the available methods
            :param subclass: the subclass to be registered
            :type subclass: object
            """
            for short_name in short_names:
                cls.available_modes[short_name] = subclass
            return subclass

        return decorator

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        conf_cluster,
        out_dir,
        log_dir,
        launch_worker=True,
        data_to_propagate=None,
    ):  # pylint: disable=W0613
        """
        Init function of AbstractCluster

        :param conf_cluster: configuration for cluster
        :param data_to_propagate: data to propagate to new cluster if reset
        :type data_to_propagate: dict

        """
        self.out_dir = out_dir

        # data to propagate
        self.data_to_propagate = data_to_propagate

        self.worker_log_dir = os.path.join(log_dir, "workers_log")
        if not os.path.exists(self.worker_log_dir):
            os.makedirs(self.worker_log_dir)

        self.log_level = logging.getLogger().getEffectiveLevel()
        handlers = logging.getLogger().handlers
        for hand in handlers:
            if "stdout" == hand.get_name():
                self.log_level = hand.level

        # Check conf
        self.checked_conf_cluster = self.check_conf(conf_cluster)

    @abstractmethod
    def get_delayed_type(self):
        """
        Get delayed type
        """

    @abstractmethod
    def cleanup(self, **kwargs):
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
            Create task builder

            :param argv: list of input arguments
            :param kwargs: list of named input arguments
            """
            wrapped_cars_profile = cars_profile_wrapper
            additionnal_kwargs = {
                "func": func,
                "func_name": func.__name__,
            }

            return self.create_task_wrapped(wrapped_cars_profile, nout=nout)(
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
    def future_iterator(self, future_list, timeout=None):
        """
        Iterator, iterating on computed futures

        :param future_list: future_list list
        :param timeout: time to wait for next job
        """


def cars_profile_wrapper(*argv, **kwargs):
    """
    Create a wrapper for cars_profile to be used in cluster tasks

    :param argv: args of func
    :param kwargs: kwargs of func

    """

    func = kwargs["func"]
    func_name = kwargs["func_name"]
    kwargs.pop("func")
    kwargs.pop("func_name")

    res = cars_profile(name=func_name)(func)(*argv, **kwargs)

    return res
