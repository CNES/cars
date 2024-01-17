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
This module contains the unseen registry class
"""

# CARS imports
from cars.orchestrator.registry.abstract_registry import (
    AbstractCarsDatasetRegistry,
)
from cars.orchestrator.registry.replacer_registry import (
    SingleCarsDatasetReplacer,
)


class CarsDatasetRegistryUnseen(AbstractCarsDatasetRegistry):
    """
    CarsDatasetRegistryUnseen
    This registry manages the unseen CarsDataset, that might be needed
    to get infos
    """

    def __init__(self, id_generator):
        """
        Init function of CarsDatasetRegistryUnseen

        :param id_generator: id generator
        :type id_generator: IdGenerator

        """
        super().__init__(id_generator)
        self.registered_cars_datasets_unseen = []

    def get_cars_ds(self, future_result):
        """
        Get a list of registered CarsDataset

        :param obj: object to get cars dataset from

        :return corresponding CarsDataset
        :rtype: CarsDataset
        """

        return None

    def cars_dataset_in_registry(self, cars_ds):
        """
        Check if a CarsDataset is already registered, return id if exists

        :param cars_ds: cars dataset
        :type cars_ds: CarsDataset

        :return : True if in registry, if of cars dataset
        :rtype : Tuple(bool, int)
        """

        in_registry = False
        registered_id = None
        for obj in self.registered_cars_datasets_unseen:
            if cars_ds == obj.cars_ds:
                in_registry = True
                registered_id = obj.obj_id
                break

        return in_registry, registered_id

    def add_cars_ds_to_unseen(self, cars_ds):
        """
        Add cars dataset to unseen registry, and
        get corresponding id

        :param cars_ds: cars dataset
        :type cars_ds: CarsDataset

        :return : id
        :rtype : int
        """

        # Generate_id
        new_id = self.id_generator.get_new_id(cars_ds)
        # create CarsDataset replacer (same storage)
        unseen_obj = SingleCarsDatasetReplacer(cars_ds, new_id)
        self.registered_cars_datasets_unseen.append(unseen_obj)

        return new_id

    def get_cars_datasets_list(self):
        """
        Get a list of registered CarsDataset

        :return list of CarsDataset
        :rtype: list(CarsDataset)

        """
        cars_ds_list = []

        for cars_ds_saver in self.registered_cars_datasets_unseen:
            cars_ds_list.append(cars_ds_saver.cars_ds)

        return cars_ds_list
