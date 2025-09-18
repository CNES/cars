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
cars/applications/auxiliary_filling/auxiliary_filling_from_sensors_app.py
"""

import os
import shutil
import tempfile

# Third party imports
import pytest

# CARS imports
from cars.applications.auxiliary_filling import (
    auxiliary_filling_from_sensors_app as auxiliary_app,
)
from cars.core.geometry.abstract_geometry import AbstractGeometry

# CARS Tests imports
from ...helpers import (
    absolute_data_path,
    assert_same_images,
    generate_input_json,
    temporary_dir,
)

conf_0 = {
    "save_intermediate_data": False,
    "mode": "fill_nan",
    "activated": True,
    "use_mask": False,
    "texture_interpolator": "cubic",
}

conf_1 = {
    "save_intermediate_data": True,
    "mode": "full",
    "activated": True,
    "use_mask": True,
    "texture_interpolator": "linear",
}

conf_2 = {
    "save_intermediate_data": False,
    "mode": "fill_nan",
    "activated": True,
    "use_mask": True,
    "texture_interpolator": "nearest",
}


@pytest.mark.unit_tests
@pytest.mark.parametrize(
    "conf, color_reference, classification_reference",
    [
        (
            conf_0,
            "input/color_end2end_paca_aux_filling_0.tif",
            "input/classification_end2end_paca_aux_filling_0.tif",
        ),
        (
            conf_1,
            "input/color_end2end_paca_aux_filling_1.tif",
            "input/classification_end2end_paca_aux_filling_1.tif",
        ),
        (
            conf_2,
            "input/color_end2end_paca_aux_filling_2.tif",
            "input/classification_end2end_paca_aux_filling_2.tif",
        ),
    ],
)
def test_auxiliary_filling_paca(
    conf, color_reference, classification_reference
):
    """
    Test for AuxiliaryFillingFromSensors application
    """

    geo_plugin = (
        AbstractGeometry(  # pylint: disable=abstract-class-instantiated
            "SharelocGeometry", dem=None, geoid=None
        )
    )

    input_json = absolute_data_path("input/phr_paca/input.json")

    dsm_input = absolute_data_path("input/dsm_input_auxiliary_filling.tif")
    color_input = absolute_data_path("input/color_input_auxiliary_filling.tif")
    classification_input = absolute_data_path(
        "input/classification_input_auxiliary_filling.tif"
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

        auxiliary_filling_application = (
            auxiliary_app.AuxiliaryFillingFromSensors(conf)
        )

        auxiliary_filling_application.run(
            dsm_file=dsm_input,
            color_file=local_image_color,
            classif_file=local_classification_input,
            dump_dir=os.path.join(directory, "dump_dir"),
            sensor_inputs=sensor_inputs,
            pairing=pairing,
            geom_plugin=geo_plugin,
            texture_bands=["b0"],
            output_geoid=None,
        )
        # Uncomment the 2 following instructions to update reference data
        # shutil.copy2(
        #     os.path.join(local_image_color),
        #     absolute_data_path(
        #             color_reference
        #     ),
        # )
        # shutil.copy2(
        #     os.path.join(local_classification_input),
        #     absolute_data_path(
        #             classification_reference
        #     ),
        # )
        assert_same_images(
            local_image_color, absolute_data_path(color_reference)
        )
        assert_same_images(
            local_classification_input,
            absolute_data_path(classification_reference),
        )
