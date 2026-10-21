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
cars-extractroi: helper to extract region of interest from image product
"""

import argparse
import os

import numpy as np
import rasterio as rio
from affine import Affine
from rasterio.rpc import RPC
from shapely.geometry import box
from shareloc.geomodels.rpc_readers import rpc_reader
from shareloc.geomodels.rpc_writers import (
    rpc_dict_to_rio_rpcs,
    write_rio_rpc_as_rpb,
)

from cars.core.cars_logging import logger


def get_local_image_box(image_dataset):
    """
    Get image bounds in the dataset local pixel coordinate system.

    Parameters:
        image_dataset (rio.DatasetReader): Opened image dataset.

    Returns:
        shapely.geometry.Polygon: Image bounds in local pixel coordinates.
    """

    return box(0, 0, image_dataset.width, image_dataset.height)


def get_transformed_image_box(image_dataset):
    """
    Get image bounds in the dataset pixel coordinate system.

    Parameters:
        image_dataset (rio.DatasetReader): Opened image dataset.

    Returns:
        shapely.geometry.Polygon: Image bounds expressed in the same pixel
        coordinate system as the affine transform and RPC row/col results.
    """

    return box(
        min(image_dataset.bounds.left, image_dataset.bounds.right),
        min(image_dataset.bounds.bottom, image_dataset.bounds.top),
        max(image_dataset.bounds.left, image_dataset.bounds.right),
        max(image_dataset.bounds.bottom, image_dataset.bounds.top),
    )


def get_requested_box(bbx, transformer):
    """
    Get requested bounds in the dataset pixel coordinate system.

    Parameters:
        bbx (array): The bounding box in geographic coordinates.
        transformer (rio.transform.RPCTransformer): RPC transformer.

    Returns:
        tuple: Requested box as a shapely polygon and the list of projected
        corner coordinates as (row, col) tuples.
    """

    coordinates = [
        transformer.rowcol(bbx[0], bbx[1]),
        transformer.rowcol(bbx[2], bbx[1]),
        transformer.rowcol(bbx[2], bbx[3]),
        transformer.rowcol(bbx[0], bbx[3]),
    ]
    rows = [coord[0] for coord in coordinates]
    cols = [coord[1] for coord in coordinates]

    return box(min(cols), min(rows), max(cols), max(rows)), coordinates


def is_bbx_in_image(bbx, image_dataset):
    """
    Checks if the bounding box is within the image.

    Parameters:
        bbx (array): The bounding box of the image in pixel coordinates.
        image_dataset (rio.DatasetReader): Opened image dataset.

    """
    local_image_box = get_local_image_box(image_dataset)
    transformed_image_box = get_transformed_image_box(image_dataset)

    if local_image_box.intersects(bbx):
        return (
            local_image_box.contains(bbx),
            True,
            local_image_box,
            True,
        )

    return (
        transformed_image_box.contains(bbx),
        transformed_image_box.intersects(bbx),
        transformed_image_box,
        False,
    )


def get_slices_from_bbx(image_dataset, bbx, rpc_options, external_rpc=None):
    """get slices from bounding box

    Parameters:
        image_dataset (rio.DatasetReader): Opened image dataset.
        bbx (array): The bounding box of the image.
        rpc_options (dict): Options for GDALCreateRPCTransformer.

    Returns:
        tuple: The slices from the bounding box.
    """
    rpcs = image_dataset.rpcs
    if external_rpc is not None:
        rpcs = external_rpc

    transformer = rio.transform.RPCTransformer(rpcs, **rpc_options)
    requested_box, coordinates = get_requested_box(bbx, transformer)
    _, intersects, _, uses_local_coordinates = is_bbx_in_image(
        requested_box, image_dataset
    )

    if intersects and uses_local_coordinates:
        local_coordinates = np.array(
            [(coord[1], coord[0]) for coord in coordinates]
        )
    else:
        local_coordinates = np.array(
            [
                (~image_dataset.transform) * (coord[1], coord[0])
                for coord in coordinates
            ]
        )

    col_start, row_start = np.amin(local_coordinates, axis=0)
    col_stop, row_stop = np.amax(local_coordinates, axis=0)

    return (row_start, row_stop), (col_start, col_stop)


def process_image_file(  # pylint: disable=R0917
    bbx,
    input_image_path,
    output_image_path,
    rpb_file_path,
    rpc_options,
    external_rpc_file=None,
):
    """
    Processes an image file by extracting a region based on the given geometry.

    Parameters:
        region_geometry (dict): GeoJSON-like dictionary defining the region.
        input_image_path (str): Path to the input image file.
        output_image_path (str): Path to save the output image.
        rpb_file_path (str): Path to save the .RPB file.
        rpc_options (dict): Options for GDALCreateRPCTransformer.
    """
    external_rpc = None
    if external_rpc_file is not None:
        external_rpc_dict = rpc_reader(external_rpc_file)
        rio_rpcs = rpc_dict_to_rio_rpcs(external_rpc_dict)

        external_rpc = RPC(**rio_rpcs)

    with rio.open(input_image_path) as image_dataset:
        if not image_dataset.rpcs and external_rpc is None:
            raise ValueError("Image dataset has no RPCs")
        validate_bounding_box(
            bbx, image_dataset, rpc_options, external_rpc=external_rpc
        )
        rows, cols = get_slices_from_bbx(
            image_dataset, bbx, rpc_options, external_rpc=external_rpc
        )
        row_start = rows[0]
        row_stop = rows[1]
        col_start = cols[0]
        col_stop = cols[1]
        window = rio.windows.Window(
            col_off=col_start,
            row_off=row_start,
            width=(col_stop - col_start),
            height=(row_stop - row_start),
        )
        window = window.intersection(
            rio.windows.Window(0, 0, image_dataset.width, image_dataset.height)
        )
        array = image_dataset.read(window=window)
        profile = image_dataset.profile
        profile["driver"] = "GTiff"
        profile["width"] = window.width
        profile["height"] = window.height
        profile["transform"] = image_dataset.transform * Affine.translation(
            window.col_off, window.row_off
        )
        if "crs" in profile:
            del profile["crs"]
        with rio.open(output_image_path, "w", **profile) as dst:
            # write data
            dst.write(array)
            # copy rpc
            if external_rpc is not None:
                dst.rpcs = external_rpc
            else:
                dst.rpcs = image_dataset.rpcs

        if rpb_file_path is not None:
            create_rpb_file(image_dataset, rpb_file_path)


def get_human_readable_bbox(
    image_dataset,
    rpc_options,
    external_rpc=None,
    use_local_coordinates=True,
):
    """
    Get the human-readable bounding box from an image dataset.

    Parameters:
        image_dataset (rio.DatasetReader): Opened image dataset.
        rpc_options (dict): Options for GDALCreateRPCTransformer.

    Returns:
        tuple: The human-readable bounding box in the format
        (min_x, max_x, min_y, max_y).
    """
    rpcs = image_dataset.rpcs
    if external_rpc is not None:
        rpcs = external_rpc

    transformer = rio.transform.RPCTransformer(rpcs, **rpc_options)

    if use_local_coordinates:
        image_coordinates = [
            (0, 0),
            (0, image_dataset.width),
            (image_dataset.height, image_dataset.width),
            (image_dataset.height, 0),
        ]
    else:
        image_box = get_transformed_image_box(image_dataset)
        min_col, min_row, max_col, max_row = image_box.bounds
        image_coordinates = [
            (min_row, min_col),
            (min_row, max_col),
            (max_row, max_col),
            (max_row, min_col),
        ]

    human_readable_bbx = [
        transformer.xy(row, col) for row, col in image_coordinates
    ]
    # fix coordinates to precision -7 for (x, y)
    image_coords = [
        (round(coord[0], 7), round(coord[1], 7)) for coord in human_readable_bbx
    ]

    min_x = min(coord[0] for coord in image_coords)
    max_x = max(coord[0] for coord in image_coords)
    min_y = min(coord[1] for coord in image_coords)
    max_y = max(coord[1] for coord in image_coords)

    return min_x, max_x, min_y, max_y


def validate_bounding_box(bbx, image_dataset, rpc_options, external_rpc=None):
    """
    Validate the bounding box coordinates.

    Parameters:
        bbx (array): The bounding box of the image.
        image_dataset (rio.DatasetReader): Opened image dataset.
        rpc_options (dict): Options for GDALCreateRPCTransformer.
    """

    rpcs = image_dataset.rpcs
    if external_rpc is not None:
        rpcs = external_rpc

    transformer = rio.transform.RPCTransformer(rpcs, **rpc_options)
    input_box, _ = get_requested_box(bbx, transformer)
    contains, intersects, _, uses_local_coordinates = is_bbx_in_image(
        input_box, image_dataset
    )
    if not contains:
        min_x, max_x, min_y, max_y = get_human_readable_bbox(
            image_dataset,
            rpc_options,
            external_rpc=external_rpc,
            use_local_coordinates=uses_local_coordinates,
        )
        msg = (
            "Coordinates must be between "
            f"({min_x}, {min_y}) and ({max_x}, {max_y})"
        )
        if intersects:
            logger.error(msg)
        else:
            raise ValueError(
                f"Coordinates must be between "
                f"({min_x}, {min_y}) and ({max_x}, {max_y})"
            )


def create_rpb_file(image_dataset, rpb_filename):
    """
    Create and save a .RPB file from a rasterio dataset

    Parameters:
        image_dataset (rio.DatasetReader): Opened image dataset.
        rpb_filename (str): Path to save the .RPB file.
    """
    if not image_dataset.rpcs:
        raise ValueError("Image dataset has no RPCs")
    rpcs_as_dict = image_dataset.rpcs.to_dict()
    write_rio_rpc_as_rpb(rpcs_as_dict, rpb_filename)


def main():
    """
    Main cars-extractroi entrypoint
    """
    parser = argparse.ArgumentParser(
        "cars-extractroi", description="Helper to extract roi from bounding box"
    )
    parser.add_argument(
        "-il",
        type=str,
        nargs="*",
        help="Image products. Can be a list of images or "
        " pairs of (image, rpc_file)",
        required=True,
    )

    parser.add_argument(
        "-out",
        type=str,
        help="Extracts directory",
        required=True,
    )

    parser.add_argument(
        "-bbx",
        type=float,
        nargs=4,
        help="Bounding box from two points (x1, y1) and (x2, y2)",
        metavar=("x1", "y1", "x2", "y2"),
        required=True,
    )

    parser.add_argument(
        "--rpc_height",
        type=float,
        help="Constant height offset used for projection",
    )

    parser.add_argument(
        "--rpc_dem",
        type=str,
        help="Digital Elevation Model used for projection",
    )

    parser.add_argument(
        "--generate_rpb",
        action="store_true",
        help="Generate RPB file",
    )

    args = parser.parse_args()
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    rpc_options = {}

    if args.rpc_height is not None:
        rpc_options["rpc_height"] = args.rpc_height
    if args.rpc_dem is not None:
        rpc_options["rpc_dem"] = args.rpc_dem

    # parse list
    input_pairs = []
    i = 0
    while i < len(args.il):
        image_path = args.il[i]
        if i + 1 < len(args.il) and args.il[i + 1].lower().endswith(
            (".xml", ".rpc", ".rpb", ".txt", ".geom")
        ):
            rpc_path = args.il[i + 1]
            i += 2
        else:
            rpc_path = None
            i += 1
        input_pairs.append((image_path, rpc_path))

    # check first input in list to determine pipeline
    for idx, (image_path, specific_rpc_file) in enumerate(input_pairs):
        output_image_path = os.path.join(args.out, "ext_%03d.tif" % idx)
        rpb_file_path = None
        if args.generate_rpb:
            rpb_file_path = os.path.splitext(output_image_path)[0] + ".RPB"

        process_image_file(
            args.bbx,
            image_path,
            output_image_path,
            rpb_file_path,
            rpc_options,
            external_rpc_file=specific_rpc_file,
        )


if __name__ == "__main__":
    main()
