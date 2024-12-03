# !/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
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
this module contains the AuxiliaryFillingFromSensors application class.
"""

from json_checker import Checker

from cars.applications.auxiliary_filling.auxiliary_filling import (
    AuxiliaryFilling,
)


class AuxiliaryFillingFromSensors(
    AuxiliaryFilling, short_name="auxiliary_filling_from_sensors"
):
    """
    AuxiliaryFillingFromSensors Application
    """

    def __init__(self, conf=None):
        """
        Init function of AuxiliaryFillingFromSensors

        :param conf: configuration for AuxiliaryFillingFromSensors
        :return: an application_to_use object
        """

        super().__init__(conf=conf)

    def check_conf(self, conf):
        """
        Check configuration

        :param conf: configuration to check
        :type conf: dict

        :return: overloaded configuration
        :rtype: dict

        """

        # init conf
        if conf is not None:
            overloaded_conf = conf.copy()
        else:
            conf = {}
            overloaded_conf = {}

        # Overload conf
        overloaded_conf["method"] = conf.get(
            "method", "auxiliary_filling_from_sensors"
        )

        # Saving files
        overloaded_conf["save_intermediate_data"] = conf.get(
            "save_intermediate_data", False
        )

        auxiliary_filling_schema = {
            "method": str,
            "save_intermediate_data": bool,
        }

        # Check conf
        checker = Checker(auxiliary_filling_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    def run(self):
        """
        run AuxiliaryFillingFromSensors
        """

        return True
