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
This module contains the abstract registry class
"""

import logging
from abc import abstractmethod

# CARS imports
from cars.orchestrator.orchestrator_constants import (
    CARS_DATASET_KEY,
    CARS_DS_COL,
    CARS_DS_ROW,
    SAVING_INFO,
)


class AbstractCarsDatasetRegistry:
    """
    AbstractCarsDatasetRegistry
    This is the abstract class of registries, managing delayed CarsDatasets

    """

    def __init__(self, id_generator):
        """
        Init function of AbstractCarsDatasetRegistry

        :param id_generator: id generator
        :type id_generator: IdGenerator

        """

        self.id_generator = id_generator
        self.id_generator.add_registry(self)

    @abstractmethod
    def cars_dataset_in_registry(self, cars_ds):
        """
        Check if a CarsDataset is already registered, return id if exists

        :param cars_ds: cars dataset
        :type cars_ds: CarsDataset

        :return : True if in registry, if of cars dataset
        :rtype : Tuple(bool, int)
        """

    @abstractmethod
    def get_cars_datasets_list(self):
        """
        Get a list of registered CarsDataset

        :return list of CarsDataset
        :rtype: list(CarsDataset)
        """

    @abstractmethod
    def get_cars_ds(self, future_result):
        """
        Get a list of registered CarsDataset

        :param future_result: object to get cars dataset from

        :return corresponding CarsDataset
        :rtype: CarsDataset
        """

    @staticmethod
    def get_future_cars_dataset_id(future_result):
        """
        Get cars dataset id for current future result

        :param future_result: future result:
        :type future_result: xr.Dataset or pd.DataFrame

        :return cars dataset id
        :rtype : int
        """

        cars_ds_id = None

        if isinstance(future_result, dict):
            attributes_info_dict = future_result
        else:
            attributes_info_dict = future_result.attrs

        if SAVING_INFO in attributes_info_dict:
            if CARS_DATASET_KEY in attributes_info_dict[SAVING_INFO]:
                cars_ds_id = attributes_info_dict[SAVING_INFO][CARS_DATASET_KEY]

        return cars_ds_id

    @staticmethod
    def get_future_cars_dataset_position(future_result):
        """
        Get cars dataset positions for current future result

        :param future_result: future result:
        :type future_result: xr.Dataset or pd.DataFrame

        :return cars dataset id
        :rtype : tuple(int)
        """

        cars_ds_row = None
        cars_ds_col = None

        if isinstance(future_result, dict):
            attributes_info_dict = future_result
        else:
            attributes_info_dict = future_result.attrs

        if SAVING_INFO in attributes_info_dict:
            if CARS_DS_ROW in attributes_info_dict[SAVING_INFO]:
                cars_ds_row = attributes_info_dict[SAVING_INFO][CARS_DS_ROW]
            else:
                logging.debug("No row given in object")
            if CARS_DS_COL in attributes_info_dict[SAVING_INFO]:
                cars_ds_col = attributes_info_dict[SAVING_INFO][CARS_DS_COL]
            else:
                logging.debug("No col given in object")

        return cars_ds_row, cars_ds_col
