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
CARS filling pipeline class file
"""

from __future__ import print_function

import warnings

import numpy as np
import rasterio
import xarray as xr
from rasterio.errors import NodataShadowWarning
from rasterio.windows import Window

from cars.data_structures import cars_dataset


def merge_filling_bands_wrapper(  # pylint: disable=R0917
    in_filling_path,
    aux_filling,
    dsm_file,
    window=None,
    saving_info=None,
    profile=None,
):
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

    with rasterio.open(in_filling_path) as src:
        nb_bands = src.count

        if nb_bands == 1:
            return False

        filling_multi_bands = src.read(window=rasterio_window)
        filling_mono_bands = np.zeros(filling_multi_bands.shape[1:3])
        descriptions = src.descriptions
        dict_temp = {name: i for i, name in enumerate(descriptions)}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NodataShadowWarning)
            filling_mask = src.read_masks(1, window=rasterio_window)

        filling_mono_bands[filling_mask == 0] = 0

        filling_bands_list = {
            "fill_with_geoid": ["filling_exogenous"],
            "interpolate_from_borders": [
                "bulldozer",
                "border_interpolation",
            ],
            "fill_with_endogenous_dem": [
                "filling_exogenous",
                "bulldozer",
            ],
            "fill_with_exogenous_dem": ["bulldozer"],
        }

        # To get the right footprint
        filling_mono_bands = np.logical_or(dsm_msk, filling_mask).astype(
            np.uint8
        )

        # to keep the previous classif convention
        filling_mono_bands[filling_mono_bands == 0] = src.nodata
        filling_mono_bands[filling_mono_bands == 1] = 0

        no_match = False
        for key, value in aux_filling.items():
            if isinstance(value, str):
                value = [value]

            if isinstance(value, list):
                for elem in value:
                    if elem != "other":
                        filling_method = filling_bands_list[elem]

                        if all(
                            method in descriptions for method in filling_method
                        ):
                            indices_true = [
                                dict_temp[m] for m in filling_method
                            ]

                            mask_true = np.all(
                                filling_multi_bands[indices_true, :, :] == 1,
                                axis=0,
                            )

                            indices_false = [
                                i
                                for i in range(filling_multi_bands.shape[0])
                                if i not in indices_true
                            ]

                            mask_false = np.all(
                                filling_multi_bands[indices_false, :, :] == 0,
                                axis=0,
                            )

                            mask = mask_true & mask_false

                            filling_mono_bands[mask] = key
                        else:
                            no_match = True

        if no_match:
            mask_1 = np.all(
                filling_multi_bands[1:, :, :] == 1,
                axis=0,
            )

            mask_2 = np.all(
                filling_mono_bands == 0,
                axis=0,
            )

            filling_mono_bands[mask_1 & mask_2] = (
                aux_filling["other"] if "other" in aux_filling else 50
            )

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
        profile=profile,
        attributes=None,
        overlaps=None,
    )

    return output_dataset


def merge_classif_bands_wrapper(  # pylint: disable=R0917
    in_classif_path,
    aux_classif,
    dsm_file,
    window=None,
    saving_info=None,
    profile=None,
):
    """
    Merge classif bands to get mono band in output
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

    with rasterio.open(in_classif_path) as src:
        nb_bands = src.count

        if nb_bands == 1:
            return False

        classif_multi_bands = src.read(window=rasterio_window)
        classif_mono_band = np.zeros(classif_multi_bands.shape[1:3])
        descriptions = src.descriptions
        classif_mask = src.read_masks(1, window=rasterio_window)
        classif_mono_band[classif_mask == 0] = 0

        # To get the right footprint
        classif_mono_band = np.logical_or(dsm_msk, classif_mask).astype(
            np.uint8
        )

        # to keep the previous classif convention
        classif_mono_band[classif_mono_band == 0] = src.nodata
        classif_mono_band[classif_mono_band == 1] = 0

        for key, value in aux_classif.items():
            if isinstance(value, int):
                num_band = descriptions.index(str(value))
                mask_1 = classif_mono_band == 0
                mask_2 = classif_multi_bands[num_band, :, :] == 1
                classif_mono_band[mask_1 & mask_2] = key
            elif isinstance(value, list):
                for elem in value:
                    num_band = descriptions.index(str(elem))
                    mask_1 = classif_mono_band == 0
                    mask_2 = classif_multi_bands[num_band, :, :] == 1
                    classif_mono_band[mask_1 & mask_2] = key

    output_dataset = xr.Dataset(
        data_vars={
            "classification": (["row", "col"], classif_mono_band),
        },
        coords={
            "row": np.arange(classif_mono_band.shape[0]),
            "col": np.arange(classif_mono_band.shape[1]),
        },
    )

    cars_dataset.fill_dataset(
        output_dataset,
        saving_info=saving_info,
        window=window,
        profile=profile,
        attributes=None,
        overlaps=None,
    )

    return output_dataset


def monoband_to_multiband_wrapper(  # pylint: disable=R0917
    input_raster,
    bands_classif,
    nodata_value,
    window=None,
    saving_info=None,
    profile=None,
):
    """
    Convert classification from monoband to multiband

    :param input_raster: the intput classification path
    :type input_raster: str
    :param bands_classif: the bands values
    :type bands_classif: list
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

    with rasterio.open(input_raster) as src:
        mono = src.read(1, window=rasterio_window)
        mono_msk = src.read_masks(1, window=rasterio_window)

    multiband = np.zeros(
        (len(bands_classif), mono.shape[0], mono.shape[1]), dtype=np.uint8
    )
    multiband_msk = np.broadcast_to(mono_msk, multiband.shape)

    for i, cls in enumerate(bands_classif):
        multiband[i] = mono == cls

    multiband[multiband_msk == 0] = nodata_value

    output_dataset = xr.Dataset(
        data_vars={
            "classification": (["band", "row", "col"], multiband),
        },
        coords={
            "band": bands_classif,
            "row": np.arange(mono.shape[0]),
            "col": np.arange(mono.shape[1]),
        },
    )

    cars_dataset.fill_dataset(
        output_dataset,
        saving_info=saving_info,
        window=window,
        profile=profile,
        attributes=None,
        overlaps=None,
    )

    return output_dataset
