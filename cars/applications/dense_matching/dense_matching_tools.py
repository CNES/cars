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
"""
This module is responsible for the dense matching algorithms:
- thus it creates a disparity map from a pair of images
"""
# pylint: disable=too-many-lines

# Standard imports
import logging
from typing import Dict, List
import pandas

# Third party imports
import numpy as np
import pandas
import pandora
import pandora.marge
import xarray as xr
from pandora import constants as p_cst
from pandora.img_tools import check_dataset
from pandora.state_machine import PandoraMachine
from pkg_resources import iter_entry_points
from scipy import interpolate

from cars.applications import application_constants

# CARS imports
from cars.applications.dense_matching import (
    dense_matching_constants as dense_match_cst,
)
from cars.applications.point_cloud_outliers_removing import (
    outlier_removing_tools,
)
from cars.applications.triangulation import triangulation_tools
from cars.conf import mask_classes
from cars.core import constants as cst
from cars.core import constants_disparity as cst_disp
from cars.core import datasets, preprocessing, projection

def get_margins(disp_min, disp_max, corr_cfg):
    """
    Get margins for the dense matching steps

    :param disp_min: Minimum disparity
    :type disp_min: int
    :param disp_max: Maximum disparity
    :type disp_max: int
    :param corr_cfg: Correlator configuration
    :type corr_cfg: dict
    :return: margins of the matching algorithm used
    """
    return pandora.marge.get_margins(disp_min, disp_max, corr_cfg["pipeline"])


def get_masks_from_pandora(
    disp: xr.Dataset, compute_disparity_masks: bool
) -> Dict[str, np.ndarray]:
    """
    Get masks dictionary from the disparity map in output of pandora.

    :param disp: disparity map (pandora output)
    :param compute_disparity_masks: compute_disparity_masks activation status
    :return: masks dictionary
    """
    masks = {}

    # Retrieve validity mask from pandora
    # Invalid pixels in validity mask are:
    #  * Bit 0: Edge of the reference image or nodata in reference image
    #  * Bit 1: Disparity interval to explore is missing or nodata in the
    #           secondary image
    #  * Bit 6: Pixel is masked on the mask of the reference image
    #  * Bit 7: Disparity to explore is masked on the mask of the secondary
    #           image
    #  * Bit 8: Pixel located in an occlusion region
    #  * Bit 9: Fake match
    validity_mask_cropped = disp.validity_mask.values
    # Mask initialization to false (all is invalid)
    masks[cst_disp.VALID] = np.full(validity_mask_cropped.shape, False)
    # Identify valid points
    masks[cst_disp.VALID][
        np.where((validity_mask_cropped & p_cst.PANDORA_MSK_PIXEL_INVALID) == 0)
    ] = True

    # With compute_disparity_masks, produce one mask for each invalid flag in
    if compute_disparity_masks:
        msk_table = dense_match_cst.MASK_HASH_TABLE
        for key, val in msk_table.items():
            masks[key] = np.full(validity_mask_cropped.shape, False)
            masks[key][np.where((validity_mask_cropped & val) == 0)] = True

    # Build final mask with 255 for valid points and 0 for invalid points
    # The mask is used by rasterize method (non zero are valid points)
    for key, mask in masks.items():
        final_msk = np.ndarray(mask.shape, dtype=np.int16)
        final_msk[mask] = 255
        final_msk[np.equal(mask, False)] = 0
        masks[key] = final_msk

    return masks


def add_color(
    output_dataset: xr.Dataset,
    color: np.ndarray = None,
    color_mask: np.ndarray = None,
):
    """ "
    Add color and color mask to dataset

    :param output_dataset: output dataset
    :param color: color array
    :param color_mask: color mask array

    """

    if color is not None:
        nb_bands = 1
        if len(color.shape) > 2:
            nb_bands = color.shape[0]

        if nb_bands > 1 and cst.BAND not in output_dataset.dims:
            output_dataset.assign_coords({cst.BAND: np.arange(nb_bands)})
            output_dataset[cst.EPI_COLOR] = xr.DataArray(
                color,
                dims=[cst.BAND, cst.ROW, cst.COL],
            )
        else:
            output_dataset[cst.EPI_COLOR] = xr.DataArray(
                color,
                dims=[cst.ROW, cst.COL],
            )

    # Add color mask
    if color_mask is not None:
        output_dataset[cst.EPI_COLOR_MSK] = xr.DataArray(
            color_mask,
            dims=[cst.ROW, cst.COL],
        )


def create_disp_dataset(
    disp: xr.Dataset,
    ref_dataset: xr.Dataset,
    compute_disparity_masks: bool = False,
) -> xr.Dataset:
    """
    Create the disparity dataset.

    :param disp: disparity map (result of pandora)
    :param ref_dataset: reference dataset for the considered disparity map
    :param compute_disparity_masks: compute_disparity_masks activation status
    :return: disparity dataset as used in cars
    """
    # Retrieve disparity values
    disp_map = disp.disparity_map.values

    # retrieve masks
    masks = get_masks_from_pandora(disp, compute_disparity_masks)

    # retrieve colors
    color = None
    nb_bands = 1
    if cst.EPI_COLOR in ref_dataset:
        color = ref_dataset[cst.EPI_COLOR].values
        if len(color.shape) > 2:
            nb_bands = color.shape[0]
            if nb_bands == 1:
                color = color[0, :, :]

    color_mask = None
    if cst.EPI_COLOR_MSK in ref_dataset:
        color_mask = ref_dataset[cst.EPI_COLOR_MSK].values

    # Crop disparity to ROI
    ref_roi = [
        int(-ref_dataset.attrs[cst.EPI_MARGINS][0]),
        int(-ref_dataset.attrs[cst.EPI_MARGINS][1]),
        int(ref_dataset.dims[cst.COL] - ref_dataset.attrs[cst.EPI_MARGINS][2]),
        int(ref_dataset.dims[cst.ROW] - ref_dataset.attrs[cst.EPI_MARGINS][3]),
    ]
    # Crop disparity map
    disp_map = disp_map[ref_roi[1] : ref_roi[3], ref_roi[0] : ref_roi[2]]

    # Crop color
    if color is not None:
        if nb_bands == 1:
            color = color[ref_roi[1] : ref_roi[3], ref_roi[0] : ref_roi[2]]
        else:
            color = color[:, ref_roi[1] : ref_roi[3], ref_roi[0] : ref_roi[2]]
    # Crop color mask
    if color_mask is not None:
        color_mask = color_mask[
            ref_roi[1] : ref_roi[3], ref_roi[0] : ref_roi[2]
        ]

    # Crop masks
    for key in masks.copy():
        masks[key] = masks[key][
            ref_roi[1] : ref_roi[3], ref_roi[0] : ref_roi[2]
        ]

    # Fill disparity array with 0 value for invalid points
    disp_map[masks[cst_disp.VALID] == 0] = 0

    # Build output dataset
    row = np.array(
        range(ref_dataset.attrs[cst.ROI][1], ref_dataset.attrs[cst.ROI][3])
    )
    col = np.array(
        range(ref_dataset.attrs[cst.ROI][0], ref_dataset.attrs[cst.ROI][2])
    )

    disp_ds = xr.Dataset(
        {
            cst_disp.MAP: ([cst.ROW, cst.COL], np.copy(disp_map)),
            cst_disp.VALID: (
                [cst.ROW, cst.COL],
                np.copy(masks[cst_disp.VALID]),
            ),
        },
        coords={cst.ROW: row, cst.COL: col},
    )

    # add color
    add_color(disp_ds, color=color, color_mask=color_mask)

    # add ambiguity_confidence
    add_ambiguity(disp_ds, disp, ref_roi)

    if compute_disparity_masks:
        for key, val in masks.items():
            disp_ds[key] = xr.DataArray(np.copy(val), dims=[cst.ROW, cst.COL])

    disp_ds.attrs = disp.attrs.copy()
    disp_ds.attrs[cst.ROI] = ref_dataset.attrs[cst.ROI]

    disp_ds.attrs[cst.EPI_FULL_SIZE] = ref_dataset.attrs[cst.EPI_FULL_SIZE]

    return disp_ds


def add_ambiguity(
    output_dataset: xr.Dataset,
    disp: xr.Dataset,
    ref_roi: List[int],
):
    """ "
    Add ambiguity to dataset

    :param output_dataset: output dataset
    :param disp: disp xarray

    """
    confidence_measure_indicator_index = list(disp.confidence_measure.indicator)
    if "ambiguity_confidence" in confidence_measure_indicator_index:
        ambiguity_idx = list(disp.confidence_measure.indicator).index(
            "ambiguity_confidence"
        )
        output_dataset[cst_disp.AMBIGUITY_CONFIDENCE] = xr.DataArray(
            np.copy(
                disp.confidence_measure.data[
                    ref_roi[1] : ref_roi[3],
                    ref_roi[0] : ref_roi[2],
                    ambiguity_idx,
                ]
            ),
            dims=[cst.ROW, cst.COL],
        )


def compute_mask_to_use_in_pandora(
    dataset: xr.Dataset,
    msk_key: str,
    classes_to_ignore: List[int],
    out_msk_dtype: np.dtype = np.int16,
) -> np.ndarray:
    """
    Compute the mask to use in Pandora.
    Valid pixels will be set to the value of the 'valid_pixels' field of the
    correlation configuration file. No data pixels will be set to the value of
    the 'no_data' field of the correlation configuration file. Nonvalid pixels
    will be set to a value automatically determined to be different from the
    'valid_pixels' and the 'no_data' fields of the correlation configuration
    file.

    :param dataset: dataset containing the multi-classes mask from which the
                    mask to used in Pandora will be computed
    :param msk_key: key to use to access the multi-classes mask in the dataset
    :param classes_to_ignore:
    :param out_msk_dtype: numpy dtype of the returned mask
    :return: the mask to use in Pandora
    """

    ds_values_list = [key for key, _ in dataset.items()]
    if msk_key not in ds_values_list:
        worker_logger = logging.getLogger("distributed.worker")
        worker_logger.fatal(
            "No value identified by {} is "
            "present in the dataset".format(msk_key)
        )
        raise Exception(
            "No value identified by {} is "
            "present in the dataset".format(msk_key)
        )

    # retrieve specific values from datasets
    # Valid values and nodata values
    valid_pixels = dataset.attrs[cst.EPI_VALID_PIXELS]
    nodata_pixels = dataset.attrs[cst.EPI_NO_DATA_MASK]

    info_dtype = np.iinfo(out_msk_dtype)

    # find a value to use for invalid pixels
    unvalid_pixels = None
    for i in range(info_dtype.max):
        if i not in (valid_pixels, nodata_pixels):
            unvalid_pixels = i
            break

    # initialization of the mask to use in Pandora
    final_msk = np.full(
        dataset[msk_key].values.shape,
        dtype=out_msk_dtype,
        fill_value=valid_pixels,
    )

    # retrieve the invalid and nodata pixels locations
    unvalid_pixels_mask = mask_classes.create_msk_from_classes(
        dataset[msk_key].values, classes_to_ignore, out_msk_dtype=bool
    )
    nodata_pixels_mask = mask_classes.create_msk_from_classes(
        dataset[msk_key].values, [nodata_pixels], out_msk_dtype=bool
    )

    # update the mask to use in pandora with the invalid and
    # nodata pixels values
    final_msk = np.where(unvalid_pixels_mask, unvalid_pixels, final_msk)
    final_msk = np.where(nodata_pixels_mask, nodata_pixels, final_msk)

    return final_msk


def compute_disparity(
    left_dataset,
    right_dataset,
    corr_cfg,
    disp_min=None,
    disp_max=None,
    mask1_ignored_by_corr=None,
    mask2_ignored_by_corr=None,
    use_sec_disp=True,
    compute_disparity_masks=False,
) -> Dict[str, xr.Dataset]:
    """
    This function will compute disparity.

    :param left_dataset: Dataset containing left image and mask
    :type left_dataset: xarray.Dataset
    :param right_dataset: Dataset containing right image and mask
    :type right_dataset: xarray.Dataset
    :param corr_cfg: Correlator configuration
    :type corr_cfg: dict
    :param disp_min: Minimum disparity
                     (if None, value is taken from left dataset)
    :type disp_min: int
    :param disp_max: Maximum disparity
                     (if None, value is taken from left dataset)
    :type disp_max: int
    :param mask1_ignored_by_corr: mask values used to ignore by correlation
    :type mask1_ignored_by_corr: List[int]
    :param mask2_ignored_by_corr: mask values used to ignore by correlation
    :type mask2_ignored_by_corr: List[int]
    :param use_sec_disp: Boolean activating the use of the secondary
                         disparity map
    :type use_sec_disp: bool
    :param compute_disparity_masks: Activation of compute_disparity_masks mode
    :type compute_disparity_masks: Boolean
    :return: Dictionary of disparity dataset. Keys are \
             (if it is computed by Pandora):

        - 'ref' for the left to right disparity map
        - 'sec' for the right to left disparity map

    """

    # Check disp min and max bounds with respect to margin used for
    # rectification

    if disp_min is None:
        disp_min = left_dataset.attrs[cst.EPI_DISP_MIN]
    else:
        if disp_min < left_dataset.attrs[cst.EPI_DISP_MIN]:
            raise ValueError(
                "disp_min ({}) is lower than disp_min used to determine "
                "margin during rectification ({})".format(
                    disp_min, left_dataset["disp_min"]
                )
            )

    if disp_max is None:
        disp_max = left_dataset.attrs[cst.EPI_DISP_MAX]
    else:
        if disp_max > left_dataset.attrs[cst.EPI_DISP_MAX]:
            raise ValueError(
                "disp_max ({}) is greater than disp_max used to determine "
                "margin during rectification ({})".format(
                    disp_max, left_dataset["disp_max"]
                )
            )

    # Load pandora plugin
    for entry_point in iter_entry_points(group="pandora.plugin"):
        entry_point.load()

    mask1_use_classes = False
    mask2_use_classes = False

    if mask1_ignored_by_corr is not None:
        left_msk = left_dataset[cst.EPI_MSK].values
        left_dataset[cst.EPI_MSK].values = compute_mask_to_use_in_pandora(
            left_dataset,
            cst.EPI_MSK,
            mask1_ignored_by_corr,
        )
        mask1_use_classes = True

    if mask2_ignored_by_corr is not None:
        right_msk = right_dataset[cst.EPI_MSK].values
        right_dataset[cst.EPI_MSK].values = compute_mask_to_use_in_pandora(
            right_dataset,
            cst.EPI_MSK,
            mask2_ignored_by_corr,
        )
        mask2_use_classes = True

    # Update nodata values
    left_dataset.attrs[cst.EPI_NO_DATA_IMG] = corr_cfg["input"]["nodata_left"]
    right_dataset.attrs[cst.EPI_NO_DATA_IMG] = corr_cfg["input"]["nodata_right"]

    # Instantiate pandora state machine
    pandora_machine = PandoraMachine()

    # check datasets
    check_dataset(left_dataset)
    check_dataset(right_dataset)

    # Run the Pandora pipeline
    ref, sec = pandora.run(
        pandora_machine,
        left_dataset,
        right_dataset,
        int(disp_min),
        int(disp_max),
        corr_cfg["pipeline"],
    )

    # Set the datasets' cst.EPI_MSK values back to the original
    # multi-classes masks
    if mask1_use_classes:
        left_dataset[cst.EPI_MSK].values = left_msk
    if mask2_use_classes:
        right_dataset[cst.EPI_MSK].values = right_msk

    disp = {}
    disp[cst.STEREO_REF] = create_disp_dataset(
        ref, left_dataset, compute_disparity_masks=compute_disparity_masks
    )

    if bool(sec.dims) and use_sec_disp:
        # for the secondary disparity map, the reference is the right dataset
        # and the secondary image is the left one
        logging.info(
            "Secondary disparity map will be used to densify "
            "the points cloud"
        )
        disp[cst.STEREO_SEC] = create_disp_dataset(
            sec,
            right_dataset,
            compute_disparity_masks=compute_disparity_masks,
        )

    return disp


def optimal_tile_size_pandora_plugin_libsgm(
    disp_min: int,
    disp_max: int,
    min_tile_size: int,
    max_tile_size: int,
    max_ram_per_worker: int,
    tile_size_rounding: int = 50,
    margin: int = 0,
) -> int:
    """
    Compute optimal tile size according to estimated memory usage
    (pandora_plugin_libsgm)
    Returned optimal tile size will be at least equal to tile_size_rounding.

    :param disp_min: Minimum disparity to explore
    :param disp_max: Maximum disparity to explore
    :param min_tile_size: Minimal tile size
    :param max_tile_size: Maximal tile size
    :param max_ram_per_worker: amount of RAM allocated per worker
    :param tile_size_rounding: Optimal tile size will be aligned to multiples\
                               of tile_size_rounding
    :param margin: margin to remove to the computed tile size
                   (as a percent of the computed tile size)
    :returns: Optimal tile size according to benchmarked memory usage
    """

    memory = max_ram_per_worker
    disp = disp_max - disp_min

    image = 32 * 2
    disp_ref = 32
    validity_mask_ref = 16
    confidence = 32
    cv_ = disp * 32
    nan_ = disp * 8
    cv_uint = disp * 8
    penal = 8 * 32 * 2
    img_crop = 32 * 2

    tot = image + disp_ref + validity_mask_ref
    tot += confidence + 2 * cv_ + nan_ + cv_uint + penal + img_crop
    import_ = 200  # MiB

    row_or_col = float(((memory - import_) * 2**23)) / tot

    if row_or_col <= 0:
        logging.warning(
            "Optimal tile size is null, "
            "forcing it to {} pixels".format(tile_size_rounding)
        )
        tile_size = tile_size_rounding
    else:
        tile_size = (1.0 - margin / 100.0) * np.sqrt(row_or_col)
        tile_size = tile_size_rounding * int(tile_size / tile_size_rounding)

    if tile_size > max_tile_size:
        tile_size = max_tile_size
    elif tile_size < min_tile_size:
        tile_size = min_tile_size

    return tile_size


def estimate_color_from_disparity(
    disp_ref_to_sec: xr.Dataset,
    sec_ds: xr.Dataset,
) -> xr.Dataset:
    """
    Estimate color image of reference from the disparity map and the secondary
    color image.

    :param disp_ref_to_sec: disparity map
    :param sec_ds: secondary image dataset
    :return: interpolated reference color image dataset
    """
    # retrieve numpy arrays from input datasets

    disp_msk = disp_ref_to_sec[cst_disp.VALID].values
    im_color = sec_ds[cst.EPI_COLOR].values
    if cst.EPI_COLOR_MSK in sec_ds.variables.keys():
        im_msk = sec_ds[cst.EPI_COLOR_MSK].values

    # retrieve image sizes
    if len(im_color.shape) == 2:
        im_color = np.expand_dims(im_color, axis=0)
    nb_bands, nb_row, nb_col = im_color.shape
    nb_disp_row, nb_disp_col = disp_ref_to_sec[cst_disp.MAP].values.shape

    sec_up_margin = abs(sec_ds.attrs[cst.EPI_MARGINS][1])
    sec_left_margin = abs(sec_ds.attrs[cst.EPI_MARGINS][0])

    # instantiate final image
    final_interp_color = np.zeros(
        (nb_disp_row, nb_disp_col, nb_bands), dtype=np.float64
    )

    # construct secondary color image pixels positions
    clr_x_positions, clr_y_positions = np.meshgrid(
        np.linspace(0, nb_col - 1, nb_col), np.linspace(0, nb_row - 1, nb_row)
    )
    clr_xy_positions = np.concatenate(
        [
            clr_x_positions.reshape(1, nb_row * nb_col).transpose(),
            clr_y_positions.reshape(1, nb_row * nb_col).transpose(),
        ],
        axis=1,
    )

    # construct the positions for which the interpolation has to be done
    interpolated_points = np.zeros(
        (nb_disp_row * nb_disp_col, 2), dtype=np.float64
    )
    for i in range(0, disp_ref_to_sec[cst_disp.MAP].values.shape[0]):
        for j in range(0, disp_ref_to_sec[cst_disp.MAP].values.shape[1]):
            # if the pixel is valid,
            # else the position is left to (0,0)
            # and the final image pixel value will be set to np.nan
            if disp_msk[i, j] == 255:
                idx = j + disp_ref_to_sec[cst_disp.MAP].values[i, j]
                interpolated_points[i * nb_disp_col + j, 0] = (
                    idx + sec_left_margin
                )
                interpolated_points[i * nb_disp_col + j, 1] = i + sec_up_margin

    # construct final image mask
    final_msk = disp_msk
    if cst.EPI_COLOR_MSK in sec_ds.variables.keys():
        # interpolate the color image mask to the new image referential
        # (nearest neighbor interpolation)
        msk_values = im_msk.reshape(nb_row * nb_col, 1)
        interp_msk_value = interpolate.griddata(
            clr_xy_positions, msk_values, interpolated_points, method="nearest"
        )
        interp_msk = interp_msk_value.reshape(nb_disp_row, nb_disp_col)

        # remove from the final mask all values which are interpolated from non
        # valid values (strictly non equal to 255)
        final_msk[interp_msk == 0] = 0

    # interpolate each band of the color image
    for band in range(nb_bands):
        # get band values
        band_im = im_color[band, :, :]
        clr_values = band_im.reshape(nb_row * nb_col, 1)

        # interpolate values
        interp_values = interpolate.griddata(
            clr_xy_positions, clr_values, interpolated_points, method="nearest"
        )
        final_interp_color[:, :, band] = interp_values.reshape(
            nb_disp_row, nb_disp_col
        )

        # apply final mask
        final_interp_color[:, :, band][final_msk != 255] = np.nan

    # create interpolated color image dataset
    region = list(disp_ref_to_sec.attrs[cst.ROI])
    largest_size = disp_ref_to_sec.attrs[cst.EPI_FULL_SIZE]

    interp_clr_ds = datasets.create_im_dataset(
        final_interp_color, region, largest_size, band_coords=True, msk=None
    )
    interp_clr_ds.attrs[cst.ROI] = disp_ref_to_sec.attrs[cst.ROI]

    return interp_clr_ds


def compute_disp_min_disp_max(
    sensor_image_right,
    sensor_image_left,
    grid_left,
    corrected_grid_right,
    grid_right,
    matches,
    orchestrator,
    geometry_loader,
    srtm_dir,
    default_alt,
    pair_folder="",
    disp_margin=0.1,
    pair_key=None,
    disp_to_alt_ratio=None,
):
    """
    Compute disp min and disp max from triangulated and filtered matches

    :param sensor_image_right: sensor image right
    :type sensor_image_right: CarsDataset
    :param sensor_image_left: sensor image left
    :type sensor_image_left: CarsDataset
    :param grid_left: grid left
    :type grid_left: CarsDataset CarsDataset
    :param corrected_grid_right: corrected grid right
    :type corrected_grid_right: CarsDataset
    :param grid_right: uncorrected grid right
    :type grid_right: CarsDataset
    :param matches: matches
    :type matches: np.ndarray
    :param orchestrator: orchestrator used
    :type orchestrator: Orchestrator
    :param geometry_loader: geometry loader to use
    :type geometry_loader: str
    :param srtm_dir: srtm directory
    :type srtm_dir: str
    :param default_alt: default altitude
    :type default_alt: float
    :param pair_folder: folder used for current pair
    :type pair_folder: str
    :param disp_margin: disparity margin
    :type disp_margin: float
    :param disp_to_alt_ratio: used for logging info
    :type disp_to_alt_ratio: float


    :return: disp min and disp max
    :rtype: float, float
    """
    input_stereo_cfg = (
        preprocessing.create_former_cars_post_prepare_configuration(
            sensor_image_left,
            sensor_image_right,
            grid_left,
            corrected_grid_right,
            pair_folder,
            uncorrected_grid_right=grid_right,
            srtm_dir=srtm_dir,
            default_alt=default_alt,
        )
    )

    point_cloud = triangulation_tools.triangulate_matches(
        geometry_loader, input_stereo_cfg, matches
    )

    # compute epsg
    epsg = preprocessing.compute_epsg(
        sensor_image_left,
        sensor_image_right,
        grid_left,
        corrected_grid_right,
        geometry_loader,
        orchestrator=orchestrator,
        pair_folder=pair_folder,
        srtm_dir=srtm_dir,
        default_alt=default_alt,
        disp_min=0,
        disp_max=0,
    )
    # Project point cloud to UTM
    projection.points_cloud_conversion_dataset(point_cloud, epsg)

    # Convert point cloud to pandas format to allow statistical filtering
    labels = [cst.X, cst.Y, cst.Z, cst.DISPARITY, cst.POINTS_CLOUD_CORR_MSK]
    cloud_array = []
    cloud_array.append(point_cloud[cst.X].data)
    cloud_array.append(point_cloud[cst.Y].data)
    cloud_array.append(point_cloud[cst.Z].data)
    cloud_array.append(point_cloud[cst.DISPARITY].data)
    cloud_array.append(point_cloud[cst.POINTS_CLOUD_CORR_MSK].data)
    pd_cloud = pandas.DataFrame(
        np.transpose(np.array(cloud_array)[:, :, 0]), columns=labels
    )

    # Statistical filtering
    filter_cloud, _ = outlier_removing_tools.statistical_outliers_filtering(
        pd_cloud, k=25, std_factor=3.0
    )

    # Obtain dmin dmax
    filt_disparity = np.array(filter_cloud.iloc[:, 3])
    dmax = np.max(filt_disparity)
    dmin = np.min(filt_disparity)

    margin = abs(dmax - dmin) * disp_margin
    dmin -= margin
    dmax += margin

    logging.info(
        "Disparity range with margin: [{:.3f} pix., {:.3f} pix.] "
        "(margin = {:.3f} pix.)".format(dmin, dmax, margin)
    )

    if disp_to_alt_ratio is not None:
        logging.info(
            "Equivalent range in meters: [{:.3f} m, {:.3f} m] "
            "(margin = {:.3f} m)".format(
                dmin * disp_to_alt_ratio,
                dmax * disp_to_alt_ratio,
                margin * disp_to_alt_ratio,
            )
        )

    # update orchestrator_out_json
    updating_infos = {
        application_constants.APPLICATION_TAG: {
            pair_key: {
                dense_match_cst.DISPARITY_RANGE_COMPUTATION_TAG: {
                    dense_match_cst.DISPARITY_MARGIN_PARAM_TAG: disp_margin,
                    dense_match_cst.MINIMUM_DISPARITY_TAG: dmin,
                    dense_match_cst.MAXIMUM_DISPARITY_TAG: dmax,
                }
            }
        }
    }
    orchestrator.update_out_info(updating_infos)

    return dmin, dmax
