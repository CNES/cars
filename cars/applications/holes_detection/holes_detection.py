# !/usr/bin/env python
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
CARS holes detection module init file
"""

import logging
from abc import ABCMeta, abstractmethod
from typing import Dict

from cars.applications.application import Application
from cars.applications.application_template import ApplicationTemplate


@Application.register("holes_detection")
class HolesDetection(ApplicationTemplate, metaclass=ABCMeta):
    """
    HolesDetection
    """

    available_applications: Dict = {}
    default_application = "cloud_to_bbox"

    def __new__(cls, conf=None):  # pylint: disable=W0613
        """
        Return the required application
        :raises:
         - KeyError when the required application is not registered

        :param conf: configuration for holes_detection
        :return: a application_to_use object
        """

        holes_detection_method = cls.default_application
        if bool(conf) is False:
            logging.info(
                "Holes Detection method not specified, "
                "default {} is used".format(holes_detection_method)
            )
        else:
            holes_detection_method = conf["method"]

        if holes_detection_method not in cls.available_applications:
            logging.error(
                "No holes_detection application named {} registered".format(
                    holes_detection_method
                )
            )
            raise KeyError(
                "No holes_detection application named {} registered".format(
                    holes_detection_method
                )
            )

        logging.info(
            "The HolesDetection({}) application will be used".format(
                holes_detection_method
            )
        )

        return super(HolesDetection, cls).__new__(
            cls.available_applications[holes_detection_method]
        )

    def __init_subclass__(cls, short_name, **kwargs):  # pylint: disable=E0302
        super().__init_subclass__(**kwargs)
        cls.available_applications[short_name] = cls

    @abstractmethod
    def run(
        self,
        epipolar_images_left,
        epipolar_images_right,
        is_activated=True,
        margin=0,
        mask_holes_to_fill_left=None,
        mask_holes_to_fill_right=None,
        orchestrator=None,
        pair_folder=None,
        pair_key="PAIR_0",
    ):
        """
        Run Refill application using plane method.

        :param epipolar_images_left:  left epipolar image
        :type epipolar_images_left: CarsDataset
        :param epipolar_images_right:  right epipolar image
        :type epipolar_images_right: CarsDataset
        :param is_activated:  activate application
        :type is_activated: bool
        :param margin: margin to use
        :type margin: int
        :param mask_holes_to_fill_left: mask classes to use
        :type mask_holes_to_fill_left: list(int)
        :param mask_holes_to_fill_right: mask classes to use
        :type mask_holes_to_fill_right: list(int)
        :param orchestrator: orchestrator used
        :type orchestrator: Orchestrator
        :param pair_folder: folder used for current pair
        :type pair_folder: str
        :param pair_key: pair id
        :type pair_key: str

        :return: left holes, right holes
        :rtype: Tuple(CarsDataset, CarsDataset)

        """
