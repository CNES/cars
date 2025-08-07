#!/usr/bin/env python
"""
cars-bundleadjustment
"""

import argparse
import copy
import json
import logging
import os
import textwrap
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio

try:
    from rpcfit import rpc_fit
except ModuleNotFoundError:
    logging.warning(
        "Module rpcfit is not installed. "
        "RPC models will not be adjusted. "
        "Run `pip install cars[bundleadjustment]` to install "
        "missing module."
    )

from affine import Affine
from scipy import interpolate, stats
from scipy.spatial import cKDTree
from shareloc.geofunctions.triangulation import n_view_triangulation
from shareloc.geomodels.geomodel import GeoModel
from shareloc.geomodels.los import LOS
from shareloc.proj_utils import coordinates_conversion

from cars.pipelines.parameters import sensor_inputs
from cars.pipelines.pipeline import Pipeline


def matches_concatenation(matches_list, pairing, nb_decimals):
    """
    Concatenate matches computed by pair: for the second and
    subsequent pairs, the first image must be included in the list
    of previous images.
    """

    matches_dataframe_list = []
    merge_on_list = [pair[0] for pair in pairing[1:]]

    # store matches as dataframe
    for matches, pair in zip(matches_list, pairing, strict=True):
        columns = [
            "col_" + pair[0],
            "row_" + pair[0],
            "col_" + pair[1],
            "row_" + pair[1],
        ]
        matches_dataframe = pd.DataFrame(matches[:, 4:8], columns=columns)
        rounded_dataframe = matches_dataframe.round(nb_decimals).add_prefix("r")
        matches_dataframe = pd.concat(
            [matches_dataframe, rounded_dataframe], axis=1
        )
        matches_dataframe_list.append(matches_dataframe)

    # aggregate dataframe with image pivot (merge_on)
    matches_dataframe = matches_dataframe_list[0]

    for matches_to_merge, merge_on in zip(
        matches_dataframe_list[1:], merge_on_list, strict=True
    ):
        # print(matches_dataframe)
        matches_dataframe = matches_dataframe.merge(
            matches_to_merge, on=["rcol_" + merge_on, "rrow_" + merge_on]
        )
        matches_dataframe["col_" + merge_on] = (
            matches_dataframe["col_" + merge_on + "_x"]
            + matches_dataframe["col_" + merge_on + "_y"]
        ) / 2
        matches_dataframe["row_" + merge_on] = (
            matches_dataframe["row_" + merge_on + "_x"]
            + matches_dataframe["row_" + merge_on + "_y"]
        ) / 2

        matches_dataframe = matches_dataframe.drop(
            [
                "col_" + merge_on + "_x",
                "col_" + merge_on + "_y",
                "row_" + merge_on + "_x",
                "row_" + merge_on + "_y",
            ],
            axis=1,
        )

    matches_dataframe = matches_dataframe.loc[
        :, ~matches_dataframe.columns.str.startswith("rcol")
    ]
    matches_dataframe = matches_dataframe.loc[
        :, ~matches_dataframe.columns.str.startswith("rrow")
    ]
    return matches_dataframe


def estimate_intersection_residues_from_matches(
    geomodels, matches_dataframe, ignored=None
):
    """
    Compute intersections from multiple matches and estimate the
    residues as the difference between the previous sensor position and
    the inverse location of the multiple views intersection.
    """

    # compute multi line intersection
    vis_list, sis_list = [], []
    for key in geomodels.keys():
        if ignored is None or key not in ignored:
            los = LOS(
                matches_dataframe[["col_" + key, "row_" + key]].values,
                geomodels[key],
            )
            vis_list.append(los.viewing_vectors)
            sis_list.append(los.starting_points)

    vis = np.dstack(vis_list)
    vis = np.swapaxes(vis, 1, 2)

    sis = np.dstack(sis_list)
    sis = np.swapaxes(sis, 1, 2)

    intersections_ecef = n_view_triangulation(sis, vis)

    in_crs = 4978
    out_crs = 4326
    intersections_wgs84 = coordinates_conversion(
        intersections_ecef, in_crs, out_crs
    )

    lon, lat, alt = (
        intersections_wgs84[:, 0],
        intersections_wgs84[:, 1],
        intersections_wgs84[:, 2],
    )

    matches_dataframe["lon"] = lon
    matches_dataframe["lat"] = lat
    matches_dataframe["alt"] = alt

    # get inverse localisation of intersection (delta)
    for key in geomodels.keys():
        row, col, _ = geomodels[key].inverse_loc(
            matches_dataframe["lon"].values.astype(float),
            matches_dataframe["lat"].values.astype(float),
            matches_dataframe["alt"].values.astype(float),
        )
        matches_dataframe["col_" + key + "_new"] = col
        matches_dataframe["row_" + key + "_new"] = row
        matches_dataframe["delta_col_" + key] = (
            matches_dataframe["col_" + key + "_new"]
            - matches_dataframe["col_" + key]
        )
        matches_dataframe["delta_row_" + key] = (
            matches_dataframe["row_" + key + "_new"]
            - matches_dataframe["row_" + key]
        )

    return matches_dataframe


def aggregate_matches_by_cell(matches_dataframe, step, min_matches):
    """
    Aggregate matches with a step computing from matches density
    matches density: footprint divided by number of matches
    to deduce a "resolution
    """

    lon_min, lon_max = list(matches_dataframe["lon"].agg(["min", "max"]))
    lat_min, lat_max = list(matches_dataframe["lat"].agg(["min", "max"]))
    res = np.sqrt(
        ((lon_max - lon_min) * (lat_max - lat_min))
        / len(matches_dataframe.index)
    )

    cell_size = float("{:0.0e}".format(step * res))
    lon_min = np.floor((lon_min / cell_size)) * cell_size
    lat_min = np.floor((lat_min / cell_size)) * cell_size
    lon_max = np.ceil((lon_max / cell_size)) * cell_size
    lat_max = np.ceil((lat_max / cell_size)) * cell_size

    matches_dataframe["lon_cell"] = (
        (matches_dataframe["lon"] / cell_size).astype(int) + 0.5
    ) * cell_size
    matches_dataframe["lat_cell"] = (
        (matches_dataframe["lat"] / cell_size).astype(int) + 0.5
    ) * cell_size

    grouped_matches = matches_dataframe.groupby(["lon_cell", "lat_cell"])
    count = grouped_matches.count()
    regular_matches = grouped_matches.median()
    regular_matches = regular_matches[count.alt > min_matches]

    return regular_matches


def plane_regression(points, values):
    """
    Deduce a plane fitting points / values
    """
    x_coords = points[:, 0].flatten()
    y_coords = points[:, 1].flatten()

    coefficient_matrix = np.array([x_coords * 0 + 1, x_coords, y_coords]).T
    ordinate = values.flatten()

    coefs, _, _, _ = np.linalg.lstsq(coefficient_matrix, ordinate, rcond=None)

    coefs_2d = np.ndarray((2, 2))
    coefs_2d[0, 0] = coefs[0]
    coefs_2d[1, 0] = coefs[1]
    coefs_2d[0, 1] = coefs[2]
    coefs_2d[1, 1] = 0.0

    return coefs_2d


def create_deformation_grid(
    images, regular_matches, interp_mode, step, aggregate_step
):
    """
    Compute the deformation grid with a defined step
    if interp_mode is True, nan is replaced by the row mean
    else the deformation is a plane deformation
    """
    old_coordinates, new_coordinates = {}, {}
    for key in images.keys():
        old_coordinates[key] = {}
        new_coordinates[key] = {}
        with rio.open(images[key]) as reader:
            height = reader.height
            width = reader.width
            transform = reader.transform

        cols_ext, rows_ext = np.meshgrid(
            np.arange(width, step=step), np.arange(height, step=step)
        )

        cols, rows = rio.transform.xy(transform, rows_ext, cols_ext)
        shapes = {"col": cols_ext.shape, "row": rows_ext.shape}
        old_coordinates[key]["col"] = cols = np.array(cols)
        old_coordinates[key]["row"] = rows = np.array(rows)
        for dimension in ["row", "col"]:
            old_coordinates[key][dimension] = old_coordinates[key][
                dimension
            ].reshape(shapes[dimension])

        points = np.array(
            (
                regular_matches["col_" + key + "_new"].to_numpy(),
                regular_matches["row_" + key + "_new"].to_numpy(),
            )
        ).T

        extrap_delta, interp_delta = {}, {}
        for dimension in ["row", "col"]:
            values = regular_matches[
                "delta_" + dimension + "_" + key
            ].to_numpy()
            coefs_2d = plane_regression(points, values)
            extrap_delta[dimension] = np.polynomial.polynomial.polyval2d(
                cols, rows, coefs_2d
            )
            extrap_delta[dimension] = extrap_delta[dimension].reshape(
                shapes[dimension]
            )
            if interp_mode:
                interp_delta[dimension] = interpolate.griddata(
                    points=points,
                    values=values,
                    xi=(cols, rows),
                    method="linear",
                )
                tree = cKDTree(points)
                coords = np.asanyarray((cols, rows)).T
                dists, __ = tree.query(coords)
                interp_delta[dimension][dists > step / aggregate_step] = np.nan
                interp_delta[dimension] = interp_delta[dimension].reshape(
                    shapes[dimension]
                )
                isnan = np.isnan(interp_delta[dimension])
                interp_delta[dimension] = rio.fill.fillnodata(
                    interp_delta[dimension], mask=~isnan, max_search_distance=5
                )
                isnan = np.isnan(interp_delta[dimension])

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", r"All-NaN (slice|axis) encountered"
                    )
                    meanrows = np.nanmedian(interp_delta[dimension], axis=1)[
                        np.newaxis
                    ].T

                interp_delta[dimension][isnan] = np.tile(
                    meanrows, (1, shapes[dimension][1])
                )[isnan]
                isnan = np.isnan(interp_delta[dimension])
                interp_delta[dimension] = rio.fill.fillnodata(
                    interp_delta[dimension], mask=~isnan
                )

        for dimension in ["row", "col"]:
            if interp_mode:
                new_coordinates[key][dimension] = (
                    old_coordinates[key][dimension] - interp_delta[dimension]
                )
            else:
                new_coordinates[key][dimension] = (
                    old_coordinates[key][dimension] - extrap_delta[dimension]
                )

    return old_coordinates, new_coordinates


def refine_rpc(geomodels, old_coordinates, new_coordinates):
    """
    Compute new rpc matching old and new coordinates
    """
    refined_rpcs = {}
    for key in geomodels.keys():
        cols, rows = (
            old_coordinates[key]["col"],
            old_coordinates[key]["row"],
        )
        new_cols, new_rows = (
            new_coordinates[key]["col"],
            new_coordinates[key]["row"],
        )
        locs_train, target_train = [], []
        for alt in [-50, 0, 500, 1000]:
            locs_train += (
                geomodels[key]
                .direct_loc_h(
                    np.ravel(rows),
                    np.ravel(cols),
                    np.full(np.prod(cols.shape), alt),
                )
                .tolist()
            )

            target_train += np.stack(
                (np.ravel(new_cols), np.ravel(new_rows)), axis=-1
            ).tolist()

        locs_train = np.array(locs_train)
        target_train = np.array(target_train)

        nanrows = np.isnan(target_train).any(axis=1)
        target_train = target_train[~nanrows]
        locs_train = locs_train[~nanrows]

        # fit on training set
        rpc_calib, __ = rpc_fit.calibrate_rpc(
            target_train,
            locs_train,
            separate=False,
            tol=1e-10,
            max_iter=20,
            method="initLcurve",
            plot=False,
            orientation="projloc",
            get_log=True,
        )
        # evaluate on training set
        rmse_err, __, __ = rpc_fit.evaluate(rpc_calib, locs_train, target_train)
        print(
            "Training set :   Mean X-RMSE {:e}     Mean Y-RMSE {:e}".format(
                *rmse_err
            )
        )

        refined_rpcs[key] = rpc_calib.to_geotiff_dict()
    return refined_rpcs


def write_rpcs_as_geom(refined_rpcs, out_dir):
    """
    Write RPCs as geomfiles
    """
    geoms_filenames = {}
    for key in refined_rpcs.keys():
        geom = os.path.join(out_dir, key + ".geom")
        with open(geom, "w", encoding="utf-8") as writer:
            for rpc_key in refined_rpcs[key]:
                try:
                    values = refined_rpcs[key][rpc_key].split()
                    for idx, value in enumerate(values):
                        line = rpc_key.lower() + "_%02d: " % idx + str(value)
                        writer.write(line + "\n")
                except AttributeError:
                    line = (
                        rpc_key.lower() + ": " + str(refined_rpcs[key][rpc_key])
                    )
                    writer.write(line + "\n")
            writer.write("type:  ossimRpcModel\n")
            writer.write("polynomial_format:  B\n")
        geoms_filenames[key] = geom
    return geoms_filenames


def new_rpcs_from_matches(
    sensors,
    config_directory,
    sparse_matching_directory,
    pairing=None,
    nb_decimals=0,
    min_matches=50,
    step=5,
    aggregate_matches=True,
    interp_mode=False,
):
    """
    Main function of cars-bundleadjustement for new RPCs estimation:
    - Retrieve matches from pairs and concatenate
    - Estimate residues by inverse location
    - Compute new RPCs
    """
    matches_list = []
    for pair in pairing:
        matches_filename = os.path.join(
            sparse_matching_directory,
            "dump_dir",
            "sparse_matching.sift",
            "_".join(pair),
            "filtered_matches.npy",
        )
        matches = np.load(matches_filename)
        matches_list.append(matches)
        print("pair: " + str(pair) + ": " + str(matches.shape[0]) + " matches")

    matches_df = matches_concatenation(matches_list, pairing, nb_decimals)

    # retrieve sensors keys
    sensors_keys = sensors.keys()

    # store geomodels
    geomodels = {}
    for key in sensors_keys:
        geomodel_filename = sensors[key]["geomodel"] = os.path.abspath(
            os.path.join(config_directory, sensors[key]["geomodel"])
        )
        geomodels[key] = GeoModel(geomodel_filename)

    matches_df = estimate_intersection_residues_from_matches(
        geomodels, matches_df
    )
    matches_df.drop_duplicates(inplace=True)

    matches_gdf = gpd.GeoDataFrame(
        matches_df,
        geometry=gpd.points_from_xy(matches_df.lon, matches_df.lat),
        crs="EPSG:4326",
    )
    matches_gdf.to_file(
        os.path.join(sparse_matching_directory, "matches.gpkg"), driver="GPKG"
    )
    matches_gdf.to_csv(os.path.join(sparse_matching_directory, "matches.csv"))

    if aggregate_matches is True:
        matches = aggregate_matches_by_cell(
            matches_df, step=step, min_matches=min_matches
        )
    else:
        matches = matches_df
        matches = matches[(np.abs(stats.zscore(matches)) < 3).all(axis=1)]

    matches_gdf = gpd.GeoDataFrame(
        matches,
        geometry=gpd.points_from_xy(matches.lon, matches.lat),
        crs="EPSG:4326",
    )
    matches_gdf.to_file(
        os.path.join(sparse_matching_directory, "aggregate_matches.gpkg"),
        driver="GPKG",
    )
    matches_gdf.to_csv(
        os.path.join(sparse_matching_directory, "aggregate_matches.csv")
    )

    images = {}
    for key in sensors_keys:
        images[key] = sensors[key]["image"]["main_file"] = os.path.abspath(
            os.path.join(config_directory, sensors[key]["image"]["main_file"])
        )

    grid_step = step * 25
    old_coords, new_coords = create_deformation_grid(
        images, matches, interp_mode, step=grid_step, aggregate_step=step
    )

    deformation_dir = os.path.join(
        sparse_matching_directory, "deformation_grids"
    )
    os.makedirs(deformation_dir, exist_ok=True)

    for key in geomodels:
        cols, rows = new_coords[key]["col"], new_coords[key]["row"]
        transform = Affine(
            grid_step, 0.0, -grid_step / 2, 0.0, grid_step, -grid_step / 2
        )

        with rio.open(
            os.path.join(deformation_dir, "positions_" + key + ".tif"),
            "w",
            driver="GTiff",
            height=cols.shape[0],
            width=cols.shape[1],
            count=2,
            dtype=cols.dtype,
            transform=transform,
        ) as writer:

            writer.write(cols, 1)
            writer.write(rows, 2)

        with rio.open(
            os.path.join(deformation_dir, "delta_" + key + ".tif"),
            "w",
            driver="GTiff",
            height=cols.shape[0],
            width=cols.shape[1],
            count=2,
            dtype=cols.dtype,
            transform=transform,
        ) as writer:

            writer.write(cols - old_coords[key]["col"], 1)
            writer.write(rows - old_coords[key]["row"], 2)

    if interp_mode is False:
        try:
            refined_rpcs = refine_rpc(geomodels, old_coords, new_coords)
        except NameError:
            logging.warning(
                "Module rpcfit is not installed. "
                "RPC models will not be adjusted. "
                "Run `pip install cars[bundleadjustment]` to install "
                "missing module."
            )
            refined_rpcs = None
        return refined_rpcs

    return None


def cars_bundle_adjustment(conf, no_run_sparse):
    """
    cars-bundleadjustement main:
    - Launch CARS to compute homologous points (run sparse matching)
    - Compute new RPCs
    """
    with open(conf, encoding="utf-8") as reader:
        conf_as_dict = json.load(reader)

    conf_dirname = os.path.dirname(conf)
    out_dir = os.path.abspath(
        os.path.join(conf_dirname, conf_as_dict["output"]["directory"])
    )

    bundle_adjustment_config = conf_as_dict["applications"].pop(
        "bundle_adjustment"
    )

    # create configuration file + launch cars sparse matching
    sparse_matching = os.path.join(out_dir, "sparse_matching")
    sparse_matching_config = copy.deepcopy(conf_as_dict)
    sparse_matching_config["inputs"]["pairing"] = bundle_adjustment_config[
        "pairing"
    ]
    sparse_matching_config["output"]["directory"] = sparse_matching
    sparse_matching_config["output"]["product_level"] = []
    sparse_matching_config["advanced"] = {}
    sparse_matching_config["advanced"]["epipolar_resolutions"] = [1]
    if "sparse_matching.sift" not in sparse_matching_config["applications"]:
        sparse_matching_config["applications"]["sparse_matching.sift"] = {}
    sparse_matching_config["applications"]["sparse_matching.sift"][
        "save_intermediate_data"
    ] = True

    sparse_matching_config["applications"]["sparse_matching.sift"][
        "decimation_factor"
    ] = 100

    sparse_matching_pipeline = Pipeline(
        "default", sparse_matching_config, conf_dirname
    )

    if no_run_sparse is False:
        sparse_matching_pipeline.run()

    # create new refined rpcs
    conf_as_dict["inputs"] = sensor_inputs.sensors_check_inputs(
        conf_as_dict["inputs"], config_json_dir=conf_dirname
    )
    separate = bundle_adjustment_config.pop("separate")
    refined_rpcs = new_rpcs_from_matches(
        conf_as_dict["inputs"]["sensors"],
        conf_dirname,
        sparse_matching,
        **bundle_adjustment_config,
    )

    if refined_rpcs is not None:
        write_rpcs_as_geom(refined_rpcs, out_dir)

    pairing_list = conf_as_dict["inputs"]["pairing"]
    if separate is False:
        pairing_list = [pairing_list]

    for pairing in pairing_list:
        # create configuration file + launch cars dense matching
        raw = os.path.join(out_dir, "raw")
        raw_config = copy.deepcopy(conf_as_dict)
        sensors_keys = conf_as_dict["inputs"]["sensors"].keys()

        if separate:
            raw_config["inputs"]["pairing"] = [pairing]
            raw_config["output"]["directory"] = "_".join([raw] + pairing)
        else:
            raw_config["inputs"]["pairing"] = pairing
            raw_config["output"]["directory"] = raw

        raw_cfg_file = raw_config["output"]["directory"] + ".json"
        with open(raw_cfg_file, "w", encoding="utf8") as json_writer:
            json.dump(raw_config, json_writer, indent=2)

        if refined_rpcs is not None:
            # create configuration file + launch cars dense matching
            refined = os.path.join(out_dir, "refined")
            refined_config = copy.deepcopy(conf_as_dict)
            sensors_keys = conf_as_dict["inputs"]["sensors"].keys()
            for key in sensors_keys:
                refined_config["inputs"]["sensors"][key]["geomodel"] = (
                    os.path.join(out_dir, key + ".geom")
                )
            if separate:
                refined_config["inputs"]["pairing"] = [pairing]
                refined_config["output"]["directory"] = "_".join(
                    [refined] + pairing
                )
            else:
                refined_config["inputs"]["pairing"] = pairing
                refined_config["output"]["directory"] = refined

            refined_cfg_file = refined_config["output"]["directory"] + ".json"
            with open(refined_cfg_file, "w", encoding="utf8") as json_writer:
                json.dump(refined_config, json_writer, indent=2)


def cli():
    """
    Command Line Interface
    """

    parser = argparse.ArgumentParser(
        "cars-bundleadjustment",
        description="Refine multiple stereo pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
This script takes a configuration file as input, similar to \
a classic configuration file for cars, by adding a \
"bundle_adjustment" \
key and its associated value:

```
  "applications": {
    "bundle_adjustment": {
      "pairing": [["key1", "key2"], ["key1", "key3"], \
["key3", "key4"]],
      "separate": true,
      "nb_decimals": 0,
      "min_matches": 50
    }
  }
```

- Parameters "pairing" and "separate" are mandatory.
- Parameters "nb_decimals" (default value: 0) and "min_matches" \
(default value: 100) are optional.

### Generation of homologous points calculated by pair

The pairs used to calculate homologous points are those declared \
by the "pairing" value in the "bundle_adjustment" application. Please \
note: for the second and subsequent pairs, the first image must be \
included in the list of previous images. In the example above, key1 of \
the second pair is contained in the first pair, key3 of the third pair \
is contained in the second pair.

### Estimation of adjustment required

Matching points are used to adjust pairs. To find homologous points common \
 to all images, the "nb_decimals" parameter is used to round off the position \
of the points to be matched. For example, if "nb_decimals" = 0, two points in \
an image are considered to be the same if they belong to the same pixel. In \
addition, measurements related to homologous points are robustified by \
calculating statistics. The "min_matches" parameter is used to set the minimum \
number of matches per zone required to calculate these statistics."""
        ),
    )
    parser.add_argument("conf", type=str, help="Configuration File")
    parser.add_argument("--no-run-sparse", action="store_true")
    args = parser.parse_args()
    cars_bundle_adjustment(**vars(args))


if __name__ == "__main__":
    cli()
