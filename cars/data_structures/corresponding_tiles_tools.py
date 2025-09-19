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
Contains functions for array reconstructions and crop for multiple tiles
"""


# Third party imports
import numpy as np
import xarray as xr


def reconstruct_data(tiles, window, overlap):  # noqa: C901
    """
    Combine list of tiles (window, overlap, xr.Dataset) as
    one full dataset

    :param tiles: list of tiles
    :type tiles: list(tuple)
    :param window: window of base tile [row min, row max, col min col max]
    :type window: list
    :param overlap: overlap of base tile [row min, row max, col min col max]
    :type overlap: list

    :return: full dataset, row min of combined, col min of combined
    :rtype: xr.Dataset, int, int

    """

    first_valid_tile = next(
        (tile for tile in tiles if tile[2] is not None), None
    )

    if first_valid_tile is None:
        return None, None, None

    list_tags = list(first_valid_tile[2].keys())

    row_min, row_max, col_min, col_max = window
    ol_row_min, ol_row_max, ol_col_min, ol_col_max = overlap

    # find whole window and new overlaps
    for tile_window, tile_overlap, _ in tiles:
        if tile_window[0] < row_min:
            row_min, ol_row_min = tile_window[0], tile_overlap[0]
        if tile_window[1] > row_max:
            row_max, ol_row_max = tile_window[1], tile_overlap[1]
        if tile_window[2] < col_min:
            col_min, ol_col_min = tile_window[2], tile_overlap[2]
        if tile_window[3] > col_max:
            col_max, ol_col_max = tile_window[3], tile_overlap[3]

    # Generate new arr

    # Compute full shape
    nb_rows = int(row_max + ol_row_max - (row_min - ol_row_min))
    nb_cols = int(col_max + ol_col_max - (col_min - ol_col_min))

    new_coords = {}
    for key in first_valid_tile[2].coords.keys():
        if key == "row":
            new_coords["row"] = range(
                int(row_min - ol_row_min), int(row_max + ol_row_max)
            )
        elif key == "col":
            new_coords["col"] = range(
                int(col_min - ol_col_min), int(col_max + ol_col_max)
            )
        elif key == "y":
            # Doesnt contain  coordinates, but pixels position
            # after reconstruction, only used in notebooks
            new_coords["y"] = range(
                int(row_min - ol_row_min), int(row_max + ol_row_max)
            )
        elif key == "x":
            # Doesnt contain  coordinates, but pixels position
            # after reconstruction, only used in notebooks
            new_coords["x"] = range(
                int(col_min - ol_col_min), int(col_max + ol_col_max)
            )
        else:
            new_coords[key] = first_valid_tile[2].coords[key]

    new_dataset = xr.Dataset(data_vars={}, coords=new_coords)

    for tag in list_tags:
        # reconstruct data
        data_type = first_valid_tile[2][tag].dtype
        data_shape = (nb_rows, nb_cols)
        dims = first_valid_tile[2][tag].dims
        if len(dims) == 3:
            nb_bands = first_valid_tile[2][tag].values.shape[0]
            data_shape = (nb_bands, nb_rows, nb_cols)
        data = np.nan * np.zeros(data_shape)

        for tile_window, tile_overlap, tile_ds in tiles:
            # only use overlaps when on the border of full image
            # row min
            if row_min == tile_window[0]:
                # is on border -> use overlap
                real_row_min = int(tile_window[0])
                getter_offset_row_min = int(0)
            else:
                real_row_min = int(tile_window[0] + ol_row_min)
                getter_offset_row_min = int(tile_overlap[0])

            # row max
            if row_max == tile_window[1]:
                # is on border -> use overlap
                real_row_max = int(tile_window[1] + ol_row_max + ol_row_min)
                getter_offset_row_max = int(0)
            else:
                real_row_max = int(tile_window[1] + ol_row_min)
                getter_offset_row_max = int(tile_overlap[1])

            # col min
            if col_min == tile_window[2]:
                # is on border -> use overlap
                real_col_min = int(tile_window[2])
                getter_offset_col_min = int(0)
            else:
                real_col_min = int(tile_window[2] + ol_col_min)
                getter_offset_col_min = int(tile_overlap[2])

            # col max
            if col_max == tile_window[3]:
                # is on border -> use overlap
                real_col_max = int(tile_window[3] + ol_col_max + ol_col_min)
                getter_offset_col_max = int(0)
            else:
                real_col_max = int(tile_window[3] + ol_col_min)
                getter_offset_col_max = int(tile_overlap[3])

            real_row_min = int(real_row_min - row_min)
            real_row_max = int(real_row_max - row_min)
            real_col_min = int(real_col_min - col_min)
            real_col_max = int(real_col_max - col_min)

            # Fill data
            if tile_ds is not None:
                tile_data = tile_ds[tag].values

                if len(tile_data.shape) == 2:
                    data[
                        real_row_min:real_row_max, real_col_min:real_col_max
                    ] = tile_data[
                        getter_offset_row_min : tile_data.shape[0]
                        - getter_offset_row_max,
                        getter_offset_col_min : tile_data.shape[1]
                        - getter_offset_col_max,
                    ]
                else:
                    data[
                        :, real_row_min:real_row_max, real_col_min:real_col_max
                    ] = tile_data[
                        :,
                        getter_offset_row_min : tile_data.shape[1]
                        - getter_offset_row_max,
                        getter_offset_col_min : tile_data.shape[2]
                        - getter_offset_col_max,
                    ]

        # add arrays to data
        new_dataset[tag] = xr.DataArray(
            data,
            dims=dims,
        ).astype(data_type)
    return new_dataset, row_min - ol_row_min, col_min - ol_col_min
