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
Test module for cars/cluster.py
"""

# Standard imports
from __future__ import absolute_import

import tempfile

# Third party imports
import pytest

# CARS imports
from cars.cluster import dask_mode

# CARS Tests imports
from tests.utils import temporary_dir


@pytest.mark.unit_tests
def test_local_dask_cluster():
    """
    Simple start and stop local cluster test
    """
    clus, client = dask_mode.start_local_cluster(4)
    dask_mode.stop_local_cluster(clus, client)


@pytest.mark.pbs_cluster_tests
def test_dask_cluster():
    """
    End to end dask_cluster management test
    """
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        clus, client = dask_mode.start_cluster(2, "00:01:00", directory)
        _ = dask_mode.get_dashboard_link(clus)
        dask_mode.stop_cluster(clus, client)
