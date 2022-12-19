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


class CarsDatasetRegistryReplacer(AbstractCarsDatasetRegistry):
    """
    CarsDatasetRegistryReplacer
    This registry manages the replacement of arriving future results
    into corresponding CarsDataset
    """

    def __init__(self, id_generator):
        """
        Init function of CarsDatasetRegistryReplacer

        :param id_generator: id generator
        :type id_generator: IdGenerator

        """
        super().__init__(id_generator)
        self.registered_cars_datasets_replacers = []

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
        for obj in self.registered_cars_datasets_replacers:
            if cars_ds == obj.cars_ds:
                in_registry = True
                registered_id = obj.obj_id
                break

        return in_registry, registered_id

    def get_cars_datasets_list(self):
        """
        Get a list of registered CarsDataset

        :return list of CarsDataset
        :rtype: list(CarsDataset)
        """

        cars_ds_list = []

        for cars_ds_replacer in self.registered_cars_datasets_replacers:
            cars_ds_list.append(cars_ds_replacer.cars_ds)

        return cars_ds_list

    def add_cars_ds_to_replace(self, cars_ds):
        """
        Add cars dataset to registry

        :param cars_ds: cars dataset
        :type cars_ds: CarsDataset

        """

        # Generate_id
        new_id = self.id_generator.get_new_id(cars_ds)
        # create CarsDataset replacer
        replacer = SingleCarsDatasetReplacer(cars_ds, new_id)
        self.registered_cars_datasets_replacers.append(replacer)

    def get_corresponding_replacer(self, future_result):
        """
        Get replacer corresponding to future result

        :param future_result: future result
        :type future_result: xr.Dataset or pandas.DataFrame

        :return replacer
        :rtype: SingleCarsDatasetReplacer
        """

        cars_ds_id = self.get_future_cars_dataset_id(future_result)

        replacer = None

        for cars_ds_replacer in self.registered_cars_datasets_replacers:
            if cars_ds_replacer.obj_id == cars_ds_id:
                replacer = cars_ds_replacer
                break

        return replacer

    def replace(self, future_result):
        """
        Replace future result

        :param future_result: xr.Dataset or pandas.DataFrame or Dict

        """

        replacer = self.get_corresponding_replacer(future_result)

        if replacer is not None:
            if not replacer.as_been_seen:
                # reset all tiles to None
                for row in range(replacer.cars_ds.shape[0]):
                    for col in range(replacer.cars_ds.shape[1]):
                        replacer.cars_ds[row, col] = None

                replacer.as_been_seen = True

            # replace tile
            row, col = self.get_future_cars_dataset_position(future_result)

            replacer.cars_ds[row, col] = future_result


class SingleCarsDatasetReplacer:  # pylint: disable=R0903
    """
    SingleCarsDatasetReplacer

    Manages the replacement of a CarsDataset
    """

    def __init__(self, cars_ds, obj_id):
        """
        Init function of CarsDatasetRegistryReplacer

        :param cars_ds: cars dataset
        :type cars_ds: CarsDataset
        :param obj_id: object id
        :type obj_id: int
        """

        self.cars_ds = cars_ds
        self.obj_id = obj_id

        self.as_been_seen = False
