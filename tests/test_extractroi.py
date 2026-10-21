#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2026 Centre National d'Etudes Spatiales (CNES).
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
"""Tests for the cars-extractroi CLI entrypoint."""

# pylint: disable=too-many-positional-arguments

import sys

import pytest
import rasterio as rio

from cars import extractroi
from tests.helpers import absolute_data_path


@pytest.mark.unit_tests
def test_extractroi_main(monkeypatch, tmp_path):
    """Fake args should be parsed and forwarded to the processing function."""
    file_left = absolute_data_path("input/phr_ventoux/left_image.tif")
    file_right = absolute_data_path("input/phr_ventoux/right_image.tif")
    calls = []

    def fake_process_image_file(
        bbx,
        input_image_path,
        output_image_path,
        rpb_file_path,
        rpc_options,
        external_rpc_file=None,
    ):
        calls.append(
            (
                bbx,
                input_image_path,
                output_image_path,
                rpb_file_path,
                rpc_options,
                external_rpc_file,
            )
        )

    monkeypatch.setattr(
        extractroi, "process_image_file", fake_process_image_file
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cars-extractroi",
            "-il",
            file_left,
            file_right,
            "-out",
            str(tmp_path),
            "-bbx",
            "1",
            "2",
            "3",
            "4",
        ],
    )

    extractroi.main()

    assert len(calls) == 2
    assert calls[0][0] == [1.0, 2.0, 3.0, 4.0]
    assert calls[0][1] == file_left
    assert calls[0][2] == str(tmp_path / "ext_000.tif")
    assert calls[0][3] is None
    assert calls[1][1] == file_right
    assert calls[1][2] == str(tmp_path / "ext_001.tif")


@pytest.mark.unit_tests
def test_extractroi_main_generate_rpb(monkeypatch, tmp_path):
    """The optional RPB generation should create the expected output path."""
    file_left = absolute_data_path("input/phr_ventoux/left_image.tif")
    calls = []

    def fake_process_image_file(
        bbx,  # pylint: disable=unused-argument
        input_image_path,  # pylint: disable=unused-argument
        output_image_path,
        rpb_file_path,
        rpc_options,  # pylint: disable=unused-argument
        external_rpc_file=None,  # pylint: disable=unused-argument
    ):
        calls.append((output_image_path, rpb_file_path))

    monkeypatch.setattr(
        extractroi, "process_image_file", fake_process_image_file
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cars-extractroi",
            "-il",
            file_left,
            "-out",
            str(tmp_path),
            "-bbx",
            "10",
            "20",
            "30",
            "40",
            "--generate_rpb",
        ],
    )

    extractroi.main()

    assert calls[0] == (
        str(tmp_path / "ext_000.tif"),
        str(tmp_path / "ext_000.RPB"),
    )


@pytest.mark.unit_tests
def test_extractroi_main_with_rpc_file_pair(monkeypatch, tmp_path):
    """
    The CLI should accept an image followed by an external RPC file in -il.
    """
    image_path = absolute_data_path("input/phr_ventoux/left_image.tif")
    rpc_path = absolute_data_path("input/phr_ventoux/left_image.geom")

    calls = []

    def fake_process_image_file(
        bbx,
        input_image_path,
        output_image_path,
        rpb_file_path,
        rpc_options,
        external_rpc_file=None,
    ):
        calls.append(
            (
                bbx,
                input_image_path,
                output_image_path,
                rpb_file_path,
                rpc_options,
                external_rpc_file,
            )
        )

    monkeypatch.setattr(
        extractroi, "process_image_file", fake_process_image_file
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cars-extractroi",
            "-il",
            image_path,
            rpc_path,
            "-out",
            str(tmp_path),
            "-bbx",
            "1",
            "2",
            "3",
            "4",
        ],
    )

    extractroi.main()

    assert len(calls) == 1
    assert calls[0][0] == [1.0, 2.0, 3.0, 4.0]
    assert calls[0][1] == image_path
    assert calls[0][2] == str(tmp_path / "ext_000.tif")
    assert calls[0][5] == rpc_path


@pytest.mark.unit_tests
def test_extractroi_process_image_file_round_trip(tmp_path):
    """
    End-to-end test of process_image_file with a real image and RPC file
    """
    image_path = absolute_data_path("input/phr_ventoux/left_image.tif")
    rpc_path = absolute_data_path("input/phr_ventoux/left_image.geom")
    output_path = tmp_path / "cropped.tif"

    # Use a small subset bounding box inside the image
    bbox = [5.1628383, 44.2283324, 5.1640, 44.2295]

    extractroi.process_image_file(
        bbox,
        image_path,
        str(output_path),
        None,
        {},
        external_rpc_file=rpc_path,
    )

    # Verify output was created and has data
    assert output_path.exists()
    with rio.open(output_path) as dst:
        cropped_data = dst.read(1)
        assert cropped_data.size > 0
        assert dst.width == 187 and dst.height == 261
        assert dst.bounds.left == pytest.approx(5311.0)
        assert dst.bounds.right == pytest.approx(5498.0)
        assert dst.bounds.top == pytest.approx(5018.0)
        assert dst.bounds.bottom == pytest.approx(5279.0)


@pytest.mark.unit_tests
def test_extractroi_validate_bounding_box_on_cropped_image():
    """A crop should be accepted when the requested ROI intersects it."""
    image_path = absolute_data_path(
        "input/phr_ventoux/crop/left_image_crop.tif"
    )
    rpc_path = absolute_data_path("input/phr_ventoux/left_image.geom")
    bbox = [
        5.193437987477119,
        44.20581260956985,
        5.196101478687786,
        44.20669089872324,
    ]

    with rio.open(image_path) as image_dataset:
        external_rpc_dict = extractroi.rpc_reader(rpc_path)
        external_rpc = rio.rpc.RPC(
            **extractroi.rpc_dict_to_rio_rpcs(external_rpc_dict)
        )

        extractroi.validate_bounding_box(
            bbox, image_dataset, {}, external_rpc=external_rpc
        )


@pytest.mark.unit_tests
def test_extractroi_process_image_file_on_cropped_image(tmp_path):
    """A crop with a non-zero pixel origin should still be extractable."""
    image_path = absolute_data_path(
        "input/phr_ventoux/crop/left_image_crop.tif"
    )
    rpc_path = absolute_data_path("input/phr_ventoux/left_image.geom")
    output_path = tmp_path / "cropped_from_crop.tif"
    bbox = [
        5.193437987477119,
        44.20581260956985,
        5.196101478687786,
        44.20669089872324,
    ]

    extractroi.process_image_file(
        bbox,
        image_path,
        str(output_path),
        None,
        {},
        external_rpc_file=rpc_path,
    )

    assert output_path.exists()
    with rio.open(output_path) as dst:
        cropped_data = dst.read(1)
        assert cropped_data.size > 0
        assert dst.width == 424
        assert dst.height == 109
        assert dst.bounds.left == pytest.approx(5050.0)
        assert dst.bounds.right == pytest.approx(5474.0)
        assert dst.bounds.top == pytest.approx(5250.0)
        assert dst.bounds.bottom == pytest.approx(5359.0)
