#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2026 Centre National d'Etudes Spatiales (CNES).
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
Abstract depth to Z fusion application.
"""

# pylint: disable=too-many-positional-arguments

from abc import ABCMeta, abstractmethod
from typing import Dict

from cars.applications.application import Application
from cars.applications.application_template import ApplicationTemplate
from cars.core.cars_logging import logger


@Application.register("depth_to_z_fusion")
class DepthToZFusion(ApplicationTemplate, metaclass=ABCMeta):
    """
    DepthToZFusion
    """

    available_applications: Dict = {}
    default_application = "anisotropic"

    def __new__(cls, conf=None):  # pylint: disable=W0613
        """
        Return the required application.
        """

        fusion_method = cls.default_application
        if bool(conf) is False or "method" not in conf:
            logger.debug(
                "Depth to Z fusion method not specified, default %s is used",
                fusion_method,
            )
        else:
            fusion_method = conf.get("method", cls.default_application)

        if fusion_method not in cls.available_applications:
            logger.error(
                "No depth to Z fusion application named %s registered",
                fusion_method,
            )
            raise KeyError(
                "No depth to Z fusion application named {} registered".format(
                    fusion_method
                )
            )

        logger.debug(
            "The DepthToZFusion(%s) application will be used",
            fusion_method,
        )

        return super(DepthToZFusion, cls).__new__(
            cls.available_applications[fusion_method]
        )

    def __init_subclass__(cls, short_name, **kwargs):  # pylint: disable=E0302
        super().__init_subclass__(**kwargs)
        cls.available_applications[short_name] = cls

    def __init__(self, conf=None):
        """
        Init function of DepthToZFusion.
        """

        super().__init__(conf=conf)

    @abstractmethod
    def run(
        self,
        point_cloud,
        orchestrator=None,
        point_cloud_dir=None,
        point_cloud_format="laz",
        dump_dir=None,
        pair_key="PAIR_0",
    ):
        """
        Run depth to Z fusion application.
        """
