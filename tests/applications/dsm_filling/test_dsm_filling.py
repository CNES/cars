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
Cars tests/bulldozer_filling  file
"""

import os
import shutil
import tempfile
from shutil import copy2  # noqa: F401 # pylint: disable=unused-import

# Third party imports
import pytest
from shapely import Polygon

# CARS imports
from cars.applications.application import Application
from cars.pipelines.parameters import output_constants
from cars.pipelines.parameters import sensor_inputs_constants as sens_cst

# CARS Tests imports
from ...helpers import (
    absolute_data_path,
    assert_same_images,
    generate_input_json,
    get_geometry_plugin,
    temporary_dir,
)


@pytest.mark.unit_tests
def test_fill_dsm():
    """
    Tests whether the dsm_filling application does its job as expected
    in cases where a roi is given and cases where it isn't.
    """

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:

        input_json = absolute_data_path(
            "input/data_gizeh_crop/configfile_crop.json"
        )

        input_dsm_base = absolute_data_path(
            "ref_output/dsm_end2end_gizeh_crop.tif"
        )
        input_app_conf = {"activated": True}

        _, input_data = generate_input_json(
            input_json, directory, "multiprocessing"
        )

        dump_dir = os.path.join(directory, "dump_dir")
        save_dir_noroi = os.path.join(directory, "save_dir_noroi")
        save_dir_roi = os.path.join(directory, "save_dir_roi")

        os.makedirs(dump_dir)
        os.makedirs(save_dir_noroi)
        os.makedirs(save_dir_roi)
        os.makedirs(
            os.path.join(save_dir_noroi, output_constants.DSM_DIRECTORY)
        )
        os.makedirs(os.path.join(save_dir_roi, output_constants.DSM_DIRECTORY))

        input_dsm_noroi = os.path.join(
            save_dir_noroi, output_constants.DSM_DIRECTORY, "dsm.tif"
        )
        input_dsm_roi = os.path.join(
            save_dir_roi, output_constants.DSM_DIRECTORY, "dsm.tif"
        )
        shutil.copyfile(input_dsm_base, input_dsm_noroi)
        shutil.copyfile(input_dsm_base, input_dsm_roi)

        inputs = input_data["inputs"]

        geometry_plugin = get_geometry_plugin(
            dem=inputs[sens_cst.INITIAL_ELEVATION][sens_cst.DEM_PATH]
        )

        dsm_filling_application = Application("dsm_filling", input_app_conf)

        roi_epsg = 4326
        roi_poly = Polygon(
            [
                [31.1320408, 29.9786511],
                [31.1328698, 29.9770122],
                [31.1347003, 29.9781863],
                [31.1334065, 29.9791590],
                [31.1320408, 29.9786511],
            ]
        )

        # first test with no roi (fill everything)
        _ = dsm_filling_application.run(
            orchestrator=None,
            initial_elevation=geometry_plugin,
            dsm_path=input_dsm_noroi,
            roi_polys=None,
            roi_epsg=None,
            output_geoid=False,
            filling_file_name=None,
            dump_dir=dump_dir,
        )

        # copy2(
        #     input_dsm_noroi,
        #     absolute_data_path(
        #         "ref_output/dsm_filling_dsm_filled_gizeh_crop_no_roi.tif"
        #     )
        # )

        assert_same_images(
            input_dsm_noroi,
            absolute_data_path(
                "ref_output/dsm_filling_dsm_filled_gizeh_crop_no_roi.tif"
            ),
        )

        # second test with an roi
        _ = dsm_filling_application.run(
            orchestrator=None,
            initial_elevation=geometry_plugin,
            dsm_path=input_dsm_roi,
            roi_polys=roi_poly,
            roi_epsg=roi_epsg,
            output_geoid=False,
            filling_file_name=None,
            dump_dir=dump_dir,
        )

        # copy2(
        #     input_dsm_roi,
        #     absolute_data_path(
        #         "ref_output/dsm_filling_dsm_filled_gizeh_crop_roi.tif"
        #     )
        # )

        assert_same_images(
            input_dsm_roi,
            absolute_data_path(
                "ref_output/dsm_filling_dsm_filled_gizeh_crop_roi.tif"
            ),
        )
