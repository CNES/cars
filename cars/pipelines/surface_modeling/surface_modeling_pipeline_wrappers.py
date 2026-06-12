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
# pylint: disable=too-many-lines
# attribute-defined-outside-init is disabled so that we can create and use
# attributes however we need, to stick to the "everything is attribute" logic
# introduced in issue#895
# pylint: disable=attribute-defined-outside-init
# pylint: disable=too-many-nested-blocks
"""
CARS surface modeling pipeline class file
"""

from __future__ import print_function

import warnings

import numpy as np
import rasterio
import xarray as xr
from rasterio.errors import NodataShadowWarning
from rasterio.windows import Window

from cars.data_structures import cars_dataset


def merge_filling_bands_wrapper(
    filling_path,
    aux_filling,
    dsm_file,
    invalidity_file,
    window=None,
    saving_info=None,
    profile_filling=None,
):  # pylint: disable=too-many-positional-arguments
    """
    Merge filling bands to get mono band in output
    """
    # Get rasterio window
    col_min = window["col_min"]
    row_min = window["row_min"]
    col_max = window["col_max"]
    row_max = window["row_max"]
    rasterio_window = Window(
        col_off=col_min,
        row_off=row_min,
        width=(col_max - col_min),
        height=(row_max - row_min),
    )

    with rasterio.open(dsm_file) as in_dsm:
        dsm_msk = in_dsm.read_masks(1, window=rasterio_window)

    with rasterio.open(invalidity_file) as src:
        invalidity_mask = src.read(window=rasterio_window)

    with rasterio.open(filling_path) as src:
        filling_multi_bands = src.read(window=rasterio_window)
        filling_mono_bands = np.zeros(filling_multi_bands.shape[1:3])
        descriptions = src.descriptions
        dict_temp = {name: i for i, name in enumerate(descriptions)}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NodataShadowWarning)
            filling_mask = src.read_masks(1, window=rasterio_window)

        filling_mono_bands[filling_mask == 0] = 0

        filling_bands_list = [
            "fill_with_endogenous_dem",
            "no_edition",
            "interpolation",
        ]

        # To get the right footprint
        filling_mono_bands = np.logical_or(dsm_msk, filling_mask).astype(
            np.uint8
        )

        for key in filling_bands_list:
            if key == "no_edition":
                mask_1 = np.all(filling_multi_bands == 0, axis=0)
                mask_2 = filling_mono_bands == 0

                filling_val = next(
                    k for k, v in aux_filling.items() if v == key
                )
                filling_mono_bands[mask_1 & mask_2] = filling_val
                continue

            if key == "interpolation":
                filling_val = next(
                    k for k, v in aux_filling.items() if v == key
                )
                filling_mono_bands[np.any(invalidity_mask == 1, axis=0)] = (
                    filling_val
                )
                continue

            if "zeros_padding" in dict_temp:
                filling_val = next(
                    k for k, v in aux_filling.items() if v == key
                )
                band_idx = dict_temp["zeros_padding"]
                mask = filling_multi_bands[band_idx, :, :] == 1
                filling_mono_bands[mask] = filling_val

        filling_mono_bands[filling_mono_bands == src.nodata] = 0

        output_dataset = xr.Dataset(
            data_vars={
                "mono_filling": (["row", "col"], filling_mono_bands),
            },
            coords={
                "row": np.arange(filling_mono_bands.shape[0]),
                "col": np.arange(filling_mono_bands.shape[1]),
            },
        )

        cars_dataset.fill_dataset(
            output_dataset,
            saving_info=saving_info,
            window=window,
            profile=profile_filling,
            attributes=None,
            overlaps=None,
        )

        return output_dataset
