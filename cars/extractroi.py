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
import logging
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


def is_bbx_in_image(bbx, image_dataset):
    """
    Checks if the bounding box is within the image.

    Parameters:
        bbx (array): The bounding box of the image.
        image_dataset (rio.DatasetReader): Opened image dataset.

    """
    img_bounds = image_dataset.bounds
    min_height = min(img_bounds.bottom, img_bounds.top)
    max_height = max(img_bounds.bottom, img_bounds.top)
    min_width = min(img_bounds.left, img_bounds.right)
    max_width = max(img_bounds.left, img_bounds.right)
    image_box = box(min_height, min_width, max_height, max_width)

    return image_box.contains(bbx), image_box.intersects(bbx), image_box


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
    coordinates = [
        transformer.rowcol(bbx[0], bbx[1]),
        transformer.rowcol(bbx[2], bbx[1]),
        transformer.rowcol(bbx[2], bbx[3]),
        transformer.rowcol(bbx[0], bbx[3]),
    ]
    coordinates = np.array(coordinates)
    (row_start, col_start) = np.amin(coordinates, axis=0)
    (row_stop, col_stop) = np.amax(coordinates, axis=0)
    rows_global = (row_start, row_stop)
    cols_global = (col_start, col_stop)

    x_origin = image_dataset.transform.c
    y_origin = image_dataset.transform.f

    rows_local = (rows_global[0] - y_origin, rows_global[1] - y_origin)
    cols_local = (cols_global[0] - x_origin, cols_global[1] - x_origin)

    return rows_local, cols_local


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


def get_human_readable_bbox(image_dataset, rpc_options, external_rpc=None):
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

    human_readable_bbx = [
        transformer.xy(0, 0),
        transformer.xy(image_dataset.height, image_dataset.width),
    ]
    # fix coordinates to precision -7 for (x, y)
    image_coords = [
        (round(coord[0], 7), round(coord[1], 7)) for coord in human_readable_bbx
    ]
    [(x_1, y_1), (x_2, y_2)] = image_coords

    min_x, max_x = min(x_1, x_2), max(x_1, x_2)
    min_y, max_y = min(y_1, y_2), max(y_1, y_2)

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
    input_box = box(
        *transformer.rowcol(bbx[0], bbx[1]), *transformer.rowcol(bbx[2], bbx[3])
    )
    contains, intersects, _ = is_bbx_in_image(input_box, image_dataset)
    if not contains:
        min_x, max_x, min_y, max_y = get_human_readable_bbox(
            image_dataset, rpc_options, external_rpc=external_rpc
        )
        msg = (
            "Coordinates must be between "
            f"({min_x}, {min_y}) and ({max_x}, {max_y})"
        )
        if intersects:
            logging.error(msg)
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
        help="Image products",
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

    # check first input in list to determine pipeline
    for idx, image_path in enumerate(args.il):
        output_image_path = os.path.join(args.out, "ext_%03d.tif" % idx)
        rpb_file_path = None
        if args.generate_rpb:
            rpb_file_path = os.path.splitext(output_image_path)[0] + ".RPB"

        process_image_file(
            args.bbx, image_path, output_image_path, rpb_file_path, rpc_options
        )


if __name__ == "__main__":
    main()
