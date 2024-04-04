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
this module contains the achievement tracker
"""

import logging

import numpy as np

from cars.orchestrator.registry.abstract_registry import (
    AbstractCarsDatasetRegistry,
)


class AchievementTracker:
    """
    AchievementTracker
    """

    def __init__(self):
        """
        Init function of AchievementTracker

        """
        self.tracked_cars_ds = []
        self.cars_ds_ids = []
        self.achievement = []

    def track(self, cars_ds, cars_ds_id):
        """
        Track cars dataset

        :param cars_ds: cars dataset to track
        :type cars_ds: CarsDataset
        :param cars_ds_id: id of cars dataset
        :type cars_ds_id: int
        """

        if cars_ds not in self.tracked_cars_ds:
            self.tracked_cars_ds.append(cars_ds)
            self.cars_ds_ids.append(cars_ds_id)
            self.achievement.append(np.zeros(cars_ds.shape, dtype=bool))

    def add_tile(self, tile):
        """
        Add finished tile

        :param tile: finished tile
        :type tile: xarray Dataset or Pandas Dataframe
        """

        try:
            self._add_tile(tile)
        except RuntimeError:
            logging.error("Error getting id in Achiement Tracker")

    def _add_tile(self, tile):
        """
        Add finished tile

        :param tile: finished tile
        :type tile: xarray Dataset or Pandas Dataframe
        """

        # Get cars dataset id
        cars_ds_id = AbstractCarsDatasetRegistry.get_future_cars_dataset_id(
            tile
        )
        if cars_ds_id is None:
            raise RuntimeError("No id in data")
        if cars_ds_id not in self.cars_ds_ids:
            raise RuntimeError("Cars ds not registered")
        index = self.cars_ds_ids.index(cars_ds_id)

        # Get position
        row, col = AbstractCarsDatasetRegistry.get_future_cars_dataset_position(
            tile
        )
        if None in (row, col):
            logging.error("None in row, col in achievement tracker")
        else:
            # update
            self.achievement[index][row, col] = 1

    def get_remaining_tiles(self):
        """
        Get remaining tiles to compute

        :return: remaining tiles
        :rtype: list(delayed)
        """

        tiles = []

        for cars_ds, achievement in zip(  # noqa: B905
            self.tracked_cars_ds, self.achievement
        ):
            for row in range(cars_ds.shape[0]):
                for col in range(cars_ds.shape[1]):
                    if (
                        not achievement[row, col]
                        and cars_ds[row, col] is not None
                    ):
                        tiles.append(cars_ds[row, col])

        return tiles
