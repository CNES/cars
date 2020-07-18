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

import pytest
import tempfile
from cars import cluster
from utils import temporary_dir


@pytest.mark.unit_tests
def test_local_cluster():
    """
    """
    clus, client = cluster.start_local_cluster(4)
    cluster.stop_local_cluster(clus, client)


@pytest.mark.pbs_cluster_tests
def test_cluster():
    """
    End to end cluster management test
    """
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:
        clus, client = cluster.start_cluster(2, "00:01:00", directory)
        link = cluster.get_dashboard_link(clus)
        cluster.stop_cluster(clus, client)
