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
This module contains the replacer registry class
"""

# CARS imports
from cars.orchestrator.registry.abstract_registry import (
    AbstractCarsDatasetRegistry,
)


class CarsDatasetRegistryCompute(AbstractCarsDatasetRegistry):
    """
    CarsDatasetRegistryCompute
    This registry manages the computation of arriving future results
    into corresponding CarsDataset
    """

    def __init__(self, id_generator):
        """
        Init function of CarsDatasetRegistryCompute

        :param id_generator: id generator
        :type id_generator: IdGenerator

        """
        super().__init__(id_generator)
        self.registered_cars_datasets = []
        self.cars_ds_ids = []

    def cars_dataset_in_registry(self, cars_ds):
        """
        Check if a CarsDataset is already registered, return id if exists

        :param cars_ds: cars dataset
        :type cars_ds: CarsDataset

        :return : True if in registry, if of cars dataset
        :rtype : Tuple(bool, int)
        """
        cars_ds_id = None
        in_registry = False

        if cars_ds in self.registered_cars_datasets:
            in_registry = True
            cars_ds_id = self.cars_ds_ids[
                self.registered_cars_datasets.index(cars_ds)
            ]

        return in_registry, cars_ds_id

    def get_cars_datasets_list(self):
        """
        Get a list of registered CarsDataset

        :return list of CarsDataset
        :rtype: list(CarsDataset)
        """

        return self.registered_cars_datasets

    def add_cars_ds_to_compute(self, cars_ds):
        """
        Add cars dataset to registry

        :param cars_ds: cars dataset
        :type cars_ds: CarsDataset

        """

        # Generate_id
        new_id = self.id_generator.get_new_id(cars_ds)
        self.cars_ds_ids.append(new_id)
        self.registered_cars_datasets.append(cars_ds)

    def get_cars_ds(self, future_result):
        """
        Get a list of registered CarsDataset

        :param future_result: object to get cars dataset from

        :return corresponding CarsDataset
        :rtype: CarsDataset
        """
        raise RuntimeError(
            "get_cars_ds shoud not be used in CarsDatasetRegistryCompute"
        )
