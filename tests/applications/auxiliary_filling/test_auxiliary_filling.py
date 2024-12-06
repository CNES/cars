#!/usr/bin/env python
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
Test module for config of
cars/applications/auxiliary_filling/auxiliary_filling_from_sensors.py
"""

import os

# Third party imports
import pytest

# CARS imports
from cars.applications.auxiliary_filling.auxiliary_filling_from_sensors import (
    AuxiliaryFillingFromSensors,
)
from cars.core.geometry.abstract_geometry import AbstractGeometry

# CARS Tests imports
# from ...helpers import get_geoid_path


@pytest.mark.unit_tests
def test_auxiliary_filling():
    """
    Test for AuxiliaryFillingFromSensors application
    """

    conf = {
        "save_intermediate_data": False,
    }

    auxiliary_filling_application = AuxiliaryFillingFromSensors(conf)

    # TODO: local paths, use test data instead
    input_dir = "/mnt/datas/cars/data_gizeh_small/input_aux_filling"
    dsm_dir = os.path.join(input_dir, "dsm")
    dsm_file = os.path.join(dsm_dir, "dsm.tif")
    color_file = os.path.join(dsm_dir, "color.tif")
    classif_file = os.path.join(dsm_dir, "classification.tif")
    out_dir = "/mnt/datas/cars/data_gizeh_small/output_aux_filling"

    # dem = "/mnt/datas/cars/data_gizeh_small/srtm_dir/N29E031_KHEOPS.tif"
    # geoid = get_geoid_path()

    geo_plugin = (
        AbstractGeometry(  # pylint: disable=abstract-class-instantiated
            "SharelocGeometry", dem=None, geoid=None
        )
    )

    sensor_inputs = {
        "input_1": {
            "color": "/mnt/datas/cars/data_gizeh_small/color1.tif",
            "geomodel": {
                "path": "/mnt/datas/cars/data_gizeh_small/img1.geom",
                "model_type": "RPC",
            },
        }
    }

    res = auxiliary_filling_application.run(
        dsm_file, color_file, classif_file, out_dir, sensor_inputs, geo_plugin
    )

    # Dummy assert for now
    assert res is not None
