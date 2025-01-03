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
import shutil
import tempfile

# Third party imports
import pytest

# CARS imports
from cars.applications.auxiliary_filling.auxiliary_filling_from_sensors import (
    AuxiliaryFillingFromSensors,
)
from cars.core.geometry.abstract_geometry import AbstractGeometry

# CARS Tests imports
from ...helpers import absolute_data_path, generate_input_json, temporary_dir


@pytest.mark.unit_tests
def test_auxiliary_filling_paca():
    """
    Test for AuxiliaryFillingFromSensors application
    """

    conf = {
        "save_intermediate_data": False,
        "mode": "fill_nan",
        "activated": True,
    }

    geo_plugin = (
        AbstractGeometry(  # pylint: disable=abstract-class-instantiated
            "SharelocGeometry", dem=None, geoid=None
        )
    )

    input_json = absolute_data_path("input/phr_paca/input.json")

    dsm_input = absolute_data_path("ref_output/dsm_end2end_paca_bulldozer.tif")
    color_input = absolute_data_path(
        "ref_output/color_end2end_paca_bulldozer.tif"
    )
    classification_input = absolute_data_path(
        "ref_output/classification_end2end_paca_bulldozer.tif"
    )

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:

        _, input_data = generate_input_json(input_json, directory, "sequential")

        sensor_inputs = input_data["inputs"]["sensors"]
        pairing = input_data["inputs"]["pairing"]

        local_image_color = os.path.join(directory, "color.tif")
        local_classification_input = os.path.join(
            directory, "classification.tif"
        )
        shutil.copyfile(color_input, local_image_color)
        shutil.copyfile(classification_input, local_classification_input)

        auxiliary_filling_application = AuxiliaryFillingFromSensors(conf)

        auxiliary_filling_application.run(
            dsm_file=dsm_input,
            color_file=local_image_color,
            classif_file=local_classification_input,
            dump_dir=os.path.join(directory, "dump_dir"),
            sensor_inputs=sensor_inputs,
            pairing=pairing,
            geom_plugin=geo_plugin,
        )
