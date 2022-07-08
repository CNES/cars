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
this module contains the abstract id generator class
"""

from cars.orchestrator.orchestrator_constants import CARS_DATASET_KEY

# CARS imports
from cars.orchestrator.registry.unseen_registry import CarsDatasetRegistryUnseen


class IdGenerator:
    """
    IdGenerator
    Creates
    """

    def __init__(self):
        """
        Init function of IdGenerator

        """
        self.current_id = 0
        self.registries = []

        # Create this registry if user gets saving infos
        # before telling orchestrator
        # he wants to save it or replace it
        self.unseen_registry = CarsDatasetRegistryUnseen(self)
        # self.unseen_registry is now in self.registries, seen Abstract init

    def add_registry(self, registry):
        """
        Add registry to self

        :param registry: registry
        :type registry: AbstractCarsDatasetRegistry
        """

        self.registries.append(registry)

    def get_new_id(self, cars_ds):
        """
        Generate new id

        :param cars_ds: cars dataset
        :type cars_ds: CarsDataset

        :return id
        :rtype: int
        """

        # Check if dataset already registered
        registered = False
        allready_registered_id = None
        for registry in self.registries:
            (
                is_registered,
                allready_registered_id,
            ) = registry.cars_dataset_in_registry(cars_ds)
            if is_registered:
                registered = True
                returned_id = allready_registered_id

        if not registered:
            returned_id = self.current_id
            self.current_id += 1

        return returned_id

    def get_saving_infos(self, cars_ds):
        """
        Get saving infos

        :param cars_ds: cars dataset
        :type cars_ds: CarsDataset

        :return saving infos
        :rtype: dict
        """

        obj_id = None
        for registry in self.registries:
            (
                is_registered,
                allready_registered_id,
            ) = registry.cars_dataset_in_registry(cars_ds)
            if is_registered:
                obj_id = allready_registered_id

        # add cars_ds to other register to create id
        obj_id = self.unseen_registry.add_cars_ds_to_unseen(cars_ds)

        infos = {CARS_DATASET_KEY: obj_id}

        return infos
