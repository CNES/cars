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
this module contains the tile profiler class
"""

import argparse
import logging
import os
import pickle

import numpy as np
import xarray as xr

COMPUTED = [0, 255, 0]
NONE_TILE = [255, 0, 0]
UNKNOWN = [0, 0, 255]

COLOR = "color"


class TileProfiler:  # pylint: disable=too-few-public-methods
    """
    TileProfiler
    """

    def __init__(self, folder, saver_registry, replacer_registry):
        """
        Init function of TileProfiler
        """

        self.folder = folder
        self.saver_registry = saver_registry
        self.replacer_registry = replacer_registry

        self.cars_ds_ids = []
        self.monitored_cars_ds = []
        self.file_names = []
        self.arrays = []

    def add_tile(self, tile_object):
        """
        Add tile to profiling
        """

        try:
            self._add_tile(tile_object)
        except Exception:
            logging.debug("Error in TileProfiler.add_tile ")

    def _add_tile(self, tile_object):
        """
        Add tile to profiling
        """

        # Get Cars Dastaset id
        cars_ds_id = self.saver_registry.get_future_cars_dataset_id(tile_object)

        if cars_ds_id not in self.cars_ds_ids:
            self.cars_ds_ids.append(cars_ds_id)

            # Get cars_ds
            cars_ds = self.saver_registry.get_cars_ds(tile_object)
            if cars_ds is None:
                cars_ds = self.replacer_registry.get_cars_ds(tile_object)
            if cars_ds is None:
                raise RuntimeError("CARS Dataset is None")
            self.monitored_cars_ds.append(cars_ds)

            # Create array
            new_arr = np.zeros((*cars_ds.shape, 3), dtype=int)
            # Update None tiles
            for tile_row in range(cars_ds.shape[0]):
                for tile_col in range(cars_ds.shape[1]):
                    if cars_ds[tile_row, tile_col] is None:
                        new_arr[tile_row, tile_col, :] = NONE_TILE
                    else:
                        new_arr[tile_row, tile_col, :] = UNKNOWN

            # Create Dataset
            tiling_grid = cars_ds.tiling_grid
            rows = (tiling_grid[:, 0, 0] + tiling_grid[:, 0, 1]) / 2
            cols = (tiling_grid[0, :, 2] + tiling_grid[0, :, 3]) / 2

            progress = ["None", "Computed", "In progress"]
            dataset = xr.Dataset(
                data_vars={
                    "color": (["row", "col", "progress"], new_arr),
                },
                coords={"progress": progress, "row": rows, "col": cols},
                attrs={
                    "x0": tiling_grid[0, 0, 0],
                    "y0": tiling_grid[0, 0, 2],
                    "dx": tiling_grid[0, 0, 3] - tiling_grid[0, 0, 2],
                    "dy": tiling_grid[0, 0, 1] - tiling_grid[0, 0, 0],
                },
            )
            self.arrays.append(dataset)

            # Create file name
            application_name = cars_ds.name
            file_name = None
            if os.path.exists(self.folder):
                file_name = os.path.join(self.folder, application_name)

            self.file_names.append(file_name)

        # Get index
        index = self.cars_ds_ids.index(cars_ds_id)

        # Get position
        row, col = self.saver_registry.get_future_cars_dataset_position(
            tile_object
        )

        # Fill corresponding array
        self.arrays[index][COLOR].values[row, col, :] = COMPUTED

        # Save
        save_pickle(self.arrays[index], self.file_names[index])


def save_pickle(dataset, file_name):
    """
    Save pickle
    """

    try:
        if file_name is not None:

            with open(file_name, "wb") as desc:  # open a text file
                pickle.dump(dataset, desc)
    except FileNotFoundError:
        logging.error("{} could not be opened".format(file_name))


def load_pickle(file_name):
    """
    Load pickle
    """
    with open(file_name, "rb") as desc:
        dataset = pickle.load(desc)

    return dataset


def main():
    """
    Main
    """

    parser = argparse.ArgumentParser(
        "cars-dashboard",
        description="Helper to monitor tiles progress",
    )
    parser.add_argument(
        "-out",
        type=str,
        help="CARS output folder to monitor",
        required=True,
    )

    args = parser.parse_args()
    cars_output = os.path.abspath(args.out)

    try:
        import plotly.graph_objects as go  # pylint: disable=import-error,C0415
        from dash import (  # pylint: disable=import-error, C0415
            Dash,
            Input,
            Output,
            callback,
            dcc,
            html,
        )

    except ModuleNotFoundError as exc:
        message = "dash / plotly not found, install cars with dev packages"
        logging.error(message)
        raise RuntimeError(message) from exc

    app = Dash(__name__, title="CARS tiles progress")

    app.layout = html.Div(
        children=[
            html.H3(
                "\n CARS output to monitor:  {}".format(cars_output),
                style={
                    "marginTop": "20px",
                    "marginBottom": "20px",
                    "textAlign": "center",
                    "color": "#01172E",
                },
            ),
            html.H1(
                "\n \n OnGoing Tiles \n \n",
                style={
                    "marginTop": "20px",
                    "marginBottom": "20px",
                    "textAlign": "center",
                    "color": "#01172E",
                },
            ),
            html.Div(
                id="tiles-output",
                style={
                    "marginTop": "20px",
                    "marginBottom": "20px",
                    "textAlign": "center",
                    "color": "#01172E",
                },
            ),
            dcc.Interval(
                id="interval-component",
                interval=2 * 1000,  # in milliseconds
                n_intervals=0,
            ),
        ]
    )

    @callback(
        Output("tiles-output", "children"),
        Input("interval-component", "n_intervals"),
    )
    def update_figures(__):
        """
        Update figures
        """

        tiles_folder = os.path.join(cars_output, "tile_processing")
        if not os.path.exists(tiles_folder):
            return None
        seen_paths = []
        tiles = []

        for name in os.listdir(tiles_folder):
            current_path = os.path.join(tiles_folder, name)
            if current_path not in seen_paths:

                dataset = load_pickle(current_path)

                tiles.append(html.H3(name))
                fig = go.Figure(
                    data=go.Image(
                        x0=dataset.attrs["x0"] + dataset.attrs["dx"] / 2,
                        y0=dataset.attrs["y0"] + dataset.attrs["dy"] / 2,
                        dx=dataset.attrs["dx"],
                        dy=dataset.attrs["dy"],
                        z=dataset["color"].values,
                    )
                )

                tiles.append(dcc.Graph(figure=fig))

        return html.Div(tiles)

    app.run(debug=True)


if __name__ == "__main__":
    main()
