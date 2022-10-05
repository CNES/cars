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
Test module for cars/orchestrator/cluster/cluster.py
"""


# Standard imports
from __future__ import absolute_import

import tempfile

import numpy as np

# Third party imports
import pytest
import xarray as xr

# CARS imports
from cars.orchestrator.cluster import abstract_cluster

# CARS Tests imports
from ...helpers import temporary_dir


def step1_mp(data):
    """
    Step 1
    """
    return data + "_step1a", data + "_step1b"


def step2_mp(data1, data2):
    """
    Step 2
    """
    return data1 + "_" + data2


def step3_mp(data):
    """
    Step 3
    """
    return data + "_step3"


# Configurations

conf_sequential = {"mode": "sequential"}

conf_mp = {"mode": "mp", "dump_to_disk": False}

conf_local_dask = {"mode": "local_dask"}

conf_pbs_dask = {
    "mode": "pbs_dask",
    "nb_workers": 2,
    "walltime": "00:01:00",
    "use_memory_logger": False,
}


@pytest.mark.unit_tests
@pytest.mark.parametrize(
    "conf", [conf_sequential, conf_local_dask, conf_mp, conf_pbs_dask]
)
def test_tasks_pipeline(conf):
    """
    Test full distributed pipeline with task creation and execution

    :param conf: distributed conf
    """

    def step1_dask(data):
        """
        Step 1
        """
        return data + "_step1a", data + "_step1b"

    def step2_dask(data1, data2):
        """
        Step 2
        """
        return data1 + "_" + data2

    def step3_dask(data):
        """
        Step 3
        """
        return data + "_step3"

    # Multiprocressing : functions must me defined outside
    # PBS dask : function can't be imported from test_cluster (not a module)
    #       Step Functions located in pytest function
    if "dask" in conf["mode"]:
        step1 = step1_dask
        step2 = step2_dask
        step3 = step3_dask
    else:
        step1 = step1_mp
        step2 = step2_mp
        step3 = step3_mp

    current_conf = conf
    # create temporary dir
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:

        # Create cluster
        cluster = abstract_cluster.AbstractCluster(  # pylint: disable=E0110
            current_conf, directory
        )

        # Create tasks
        data_list = ["bon", "jour"]

        final_delayed = []
        for data in data_list:
            delayed_1a, delayed_1b = cluster.create_task(step1, nout=2)(data)
            delayed_2 = cluster.create_task(step2, nout=1)(
                delayed_1a, delayed_1b
            )
            delayed_3 = cluster.create_task(step3, nout=1)(delayed_2)
            final_delayed.append(delayed_3)

        # compute tasks
        futures = cluster.start_tasks(final_delayed)

        futures_results = []
        for future_res in cluster.future_iterator(futures):
            futures_results.append(future_res)

        # Test

        expected_res = [
            data + "_step1a_" + data + "_step1b" + "_step3"
            for data in data_list
        ]

        for expected in expected_res:
            assert len(futures_results) == len(expected_res)
            assert expected in futures_results

        # Close cluster
        cluster.cleanup()


def step1_array(data):
    """
    Step 1
    """

    dataset1 = xr.Dataset(
        data_vars={"im": (("row", "col"), 2 * data * np.ones((3, 4)))},
        coords={
            "row": [0, 1, 2],
            "col": [0, 1, 2, 3],
        },
        attrs={"attr": "test"},
    )

    dataset2 = xr.Dataset(
        data_vars={"im": (("row", "col"), -data * np.ones((3, 4)))},
        coords={
            "row": [0, 1, 2],
            "col": [0, 1, 2, 3],
        },
        attrs={"attr": "test"},
    )

    return dataset1, dataset2


def step2_array(dataset1, dataset2):
    """
    Step 2
    """

    arr = dataset1["im"].values + dataset2["im"].values

    dataset3 = xr.Dataset(
        data_vars={"im": (("row", "col"), arr)},
        coords={
            "row": [0, 1, 2],
            "col": [0, 1, 2, 3],
        },
        attrs={"attr": "test"},
    )

    return dataset3


conf_mp_dump = {"mode": "mp", "dump_to_disk": True}


@pytest.mark.unit_tests
@pytest.mark.parametrize("conf", [conf_mp_dump])
def test_tasks_pipeline_dump_xarray(conf):
    """
    Test full distributed pipeline with task creation and execution
    with dumped_objects

    :param conf: distributed conf
    """

    current_conf = conf
    # create temporary dir
    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:

        # Create cluster
        cluster = abstract_cluster.AbstractCluster(  # pylint: disable=E0110
            current_conf, directory
        )

        # Create tasks
        data_list = [101]

        final_delayed = []
        for data in data_list:
            delayed_1a, delayed_1b = cluster.create_task(step1_array, nout=2)(
                data
            )
            delayed_2 = cluster.create_task(step2_array, nout=1)(
                delayed_1a, delayed_1b
            )
            final_delayed.append(delayed_2)

        # compute tasks
        futures = cluster.start_tasks(final_delayed)

        futures_results = []
        for future_res in cluster.future_iterator(futures):
            futures_results.append(future_res)

        # Test
        expected = xr.Dataset(
            data_vars={"im": (("row", "col"), data_list[0] * np.ones((3, 4)))},
            coords={
                "row": [0, 1, 2],
                "col": [0, 1, 2, 3],
            },
            attrs={"attr": "test"},
        )

        np.testing.assert_array_equal(
            expected["im"].values, futures_results[0]["im"].values
        )

        # Close cluster
        cluster.cleanup()
