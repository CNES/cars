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

import logging
from abc import ABCMeta, abstractmethod
from typing import Dict


class AbstractCluster(metaclass=ABCMeta):
    """
    AbstractCluster
    """

    available_modes: Dict = {}

    def __new__(  # pylint: disable=W0613
        cls, conf_cluster, out_dir, launch_worker=True
    ):
        """
        Return the required cluster
        :raises:
         - KeyError when the required cluster is not registered

        :param conf_cluster: configuration for cluster
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

        logging.info(
            "[The AbstractCluster {}  will be used".format(cluster_mode)
        )

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

    @abstractmethod
    def cleanup(self):
        """
        Cleanup cluster
        """

    @abstractmethod
    def create_task(self, func, nout=1):
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
