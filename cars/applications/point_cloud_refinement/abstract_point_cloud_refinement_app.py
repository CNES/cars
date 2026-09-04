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
Abstract point cloud refinement application.
"""

# pylint: disable=too-many-positional-arguments

from abc import ABCMeta, abstractmethod
from typing import Dict

from cars.applications.application import Application
from cars.applications.application_template import ApplicationTemplate
from cars.core.cars_logging import logger


@Application.register("point_cloud_refinement")
class PointCloudRefinement(ApplicationTemplate, metaclass=ABCMeta):
    """
    PointCloudRefinement
    """

    available_applications: Dict = {}
    default_application = "normals_guided"

    def __new__(cls, conf=None):  # pylint: disable=W0613
        """
        Return the required application.
        """

        refinement_method = cls.default_application
        if bool(conf) is False or "method" not in conf:
            logger.debug(
                "Point cloud refinement method not specified, "
                "default %s is used",
                refinement_method,
            )
        else:
            refinement_method = conf.get("method", cls.default_application)

        if refinement_method not in cls.available_applications:
            logger.error(
                "No point cloud refinement application named %s registered",
                refinement_method,
            )
            raise KeyError(
                "No point cloud refinement application named "
                "{} registered".format(refinement_method)
            )

        logger.debug(
            "The PointCloudRefinement(%s) application will be used",
            refinement_method,
        )

        return super(PointCloudRefinement, cls).__new__(
            cls.available_applications[refinement_method]
        )

    def __init_subclass__(cls, short_name, **kwargs):  # pylint: disable=E0302
        super().__init_subclass__(**kwargs)
        cls.available_applications[short_name] = cls

    def __init__(self, conf=None):
        """
        Init function of PointCloudRefinement.
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
        Run point cloud refinement application.
        """
