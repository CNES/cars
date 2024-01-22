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


def get_slices_from_bbx(src, bbx):
    """get slices from bounding box"""
    coordinates = []
    transformer = rio.transform.RPCTransformer(src.rpcs)
    for x_bbx in [bbx[0], bbx[2]]:
        for y_bbx in [bbx[1], bbx[3]]:
            coordinates.append(transformer.rowcol(x_bbx, y_bbx))
    coordinates = np.array(coordinates)
    (row_start, col_start) = np.amin(coordinates, axis=0)
    (row_stop, col_stop) = np.amax(coordinates, axis=0)
    rows = (row_start, row_stop)
    cols = (col_start, col_stop)
    return rows, cols


def create_geom_file(src, geom_filename):
    """create .geom file from rasterio dataset"""
    rpcs_as_dict = src.rpcs.to_dict()
    with open(geom_filename, "w", encoding="utf-8") as writer:
        for key in rpcs_as_dict:
            if isinstance(rpcs_as_dict[key], list):
                for idx, coef in enumerate(rpcs_as_dict[key]):
                    writer.write(key + "_%02d" % idx + ": " + str(coef))
                    writer.write("\n")
            else:
                writer.write(key + ": " + str(rpcs_as_dict[key]))
                writer.write("\n")

        writer.write("type:  ossimRpcModel\n")
        writer.write("polynomial_format:  B\n")


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

    args = parser.parse_args()
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    # check first input in list to determine pipeline
    for idx, image in enumerate(args.il):
        ext = os.path.join(args.out, "ext_%03d.tif" % idx)
        geom = os.path.splitext(ext)[0] + ".geom"

        with rio.open(image) as src:
            # read window from bbx
            rows, cols = get_slices_from_bbx(src, args.bbx)
            window = rio.windows.Window.from_slices(rows, cols)
            array = src.read(1, window=window)

            # update profile for extract
            profile = src.profile
            profile["driver"] = "GTiff"
            profile["width"] = window.width
            profile["height"] = window.height
            if "crs" in profile:
                del profile["crs"]

            translate = Affine.translation(window.col_off, window.row_off)
            profile["transform"] *= translate

            # write extract
            with rio.open(ext, "w", **profile) as dst:
                dst.write(array, 1)

            # write associated geom
            create_geom_file(src, geom)


if __name__ == "__main__":
    main()
