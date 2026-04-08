#!/usr/bin/env python  pylint: disable=too-many-lines
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
Test  pipeline
"""

import os
import tempfile

import pytest

# CARS imports
from cars.pipelines.analysis.analysis import AnalysisPipeline

# CARS Tests imports
from tests.helpers import (
    absolute_data_path,
    generate_input_json,
    temporary_dir,
)

NB_WORKERS = 2


@pytest.mark.end2end_tests
def test_pipeline_analysis_api():
    """
    Test filling pipeline
    """

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        conf_path = absolute_data_path(
            "input/phr_ventoux/input_with_color_and_classif.json"
        )

        # Generate base configuration
        _, input_conf = generate_input_json(
            conf_path,
            directory,
            "multiprocessing",
            orchestrator_parameters={
                "nb_workers": NB_WORKERS,
                "max_ram_per_worker": 500,
            },
        )

        # Run pipeline
        pipeline = AnalysisPipeline(input_conf, absolute_data_path(directory))
        pipeline.run()

        # check output report
        report_path = os.path.join(directory, "output", "report.pdf")

        assert os.path.isfile(report_path)
