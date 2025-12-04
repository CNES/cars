#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
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
this module contains the abstract dsm merging application class.
"""
import logging
from abc import ABCMeta, abstractmethod
from typing import Dict

from cars.applications.application import Application
from cars.applications.application_template import ApplicationTemplate


@Application.register("dsm_merging")
class DsmMerging(ApplicationTemplate, metaclass=ABCMeta):
    """
    DsmMerging
    """

    available_applications: Dict = {}
    default_application = "weighted_fusion"

    def __new__(cls, conf=None):  # pylint: disable=W0613
        """
        Return the required application
        :raises:
         - KeyError when the required application is not registered

        :param orchestrator: orchestrator used
        :param conf: configuration for merging
        :return: an application_to_use object
        """

        dsm_merging_method = cls.default_application
        if bool(conf) is False or "method" not in conf:
            logging.info(
                "dsm_merging method not specified, default"
                " {} is used".format(dsm_merging_method)
            )
        else:
            dsm_merging_method = conf["method"]

        if dsm_merging_method not in cls.available_applications:
            logging.error(
                "No dsm_merging application named {} registered".format(
                    dsm_merging_method
                )
            )
            raise KeyError(
                "No dsm_merging application named {} registered".format(
                    dsm_merging_method
                )
            )

        logging.info(
            "The DsmMerging {} application will be used".format(
                dsm_merging_method
            )
        )

        return super(DsmMerging, cls).__new__(
            cls.available_applications[dsm_merging_method]
        )

    def __init_subclass__(cls, short_name, **kwargs):  # pylint: disable=E0302
        super().__init_subclass__(**kwargs)
        cls.available_applications[short_name] = cls

    def __init__(self, conf=None):
        """
        Init function of DSM Merging

        :param conf: configuration
        :return: an application_to_use object
        """

        super().__init__(conf=conf)

    @abstractmethod
    def run(self):
        """
        Run dsm merging using initial elevation and the current dsm
        """
