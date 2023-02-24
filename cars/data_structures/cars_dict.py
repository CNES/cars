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
cars_dict module:

"""


class CarsDict:
    """
    CarsDict.

    Internal CARS structure for dict representation
    """

    def __init__(self, dict_data, attributes=None):
        """
        Init function of CarsDict.
        If a path is provided, restore CarsDataset saved on disk.

        :param dict_data: dictrionary to store
        :type dict_data: dict
        :param attributes: attributes
        :type attributes: dict

        """

        self.data = dict_data
        self.attrs = attributes
        if self.attrs is None:
            self.attrs = {}

    def __repr__(self):
        """
        Repr function
        :return: printable self CarsDataset
        """
        return self.custom_print()

    def __str__(self):
        """
        Str function
        :return: printable self CarsDataset
        """
        return self.custom_print()

    def custom_print(self):
        """
        Return string of self
        :return: printable self
        """

        res = str(self.__class__) + ":  \n" "data: " + str(
            self.data
        ) + "\n" + "attrs: " + str(self.attrs)
        return res
