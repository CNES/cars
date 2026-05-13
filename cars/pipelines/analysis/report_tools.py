#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
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
CARS analysis pipeline report tools module
"""
import functools
import json
import logging
import os
import os.path
import re
from math import radians
from urllib.parse import unquote, urlparse

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import yaml
from matplotlib.patches import ConnectionPatch


def make_paths_relative(html_content, output_file_path):
    """
    Transform absolute paths in the HTML content.

    :param html_content: html content
    :param output_file_path:
    :return: htm content with relative paths
    """
    pattern = r'(src|href)=["\']([^"\']+)["\']'
    base_dir = os.path.dirname(os.path.abspath(output_file_path))

    def replacer(match):
        """
        Replace absolute paths with relative paths in the HTML content.

        :param match: match
        :return: transformed string with relative path if application
        """
        attr = match.group(1)
        raw_path = match.group(2)

        parsed_url = urlparse(raw_path)
        if parsed_url.scheme == "file" or not parsed_url.scheme:
            clean_path = unquote(parsed_url.path)
            if os.path.isabs(clean_path):
                try:
                    rel_path = os.path.relpath(clean_path, base_dir)
                    rel_path = rel_path.replace(os.sep, "/")
                    return f'{attr}="{rel_path}"'  # noqa: B907
                except ValueError:
                    return match.group(0)

        return match.group(0)

    return re.sub(pattern, replacer, html_content)


def merge_reports(reports_html_files, report_file_html, report_file_pdf):
    """
    Merge the reports pdfs into one report pdf

    :param reports_html_files: list of paths to the reports htmls to merge
    :type reports_html_files: list of str
    :param report_file_html: path to the merged report html
    :type report_file_html: str
    :param report_file_pdf: path to the merged report pdf
    :type report_file_pdf: str
    """

    # Concat all HTML strings
    html_data = []
    for html_file in reports_html_files:
        with open(html_file, "r", encoding="utf-8") as f:
            html_data.append(f.read())
    merged_html = "".join(html_data)

    # Change absolute paths to relative paths
    relative_merged_html = make_paths_relative(merged_html, report_file_html)

    # Save html report
    with open(report_file_html, "w", encoding="utf-8") as f:
        f.write(relative_merged_html)

    # Save PDF report if possible
    try:
        from weasyprint import HTML  # pylint: disable=import-outside-toplevel

        # Export to PDF
        HTML(string=merged_html).write_pdf(report_file_pdf)
    except ImportError:
        logging.warning(
            "WeasyPrint not installed, skipping PDF generation "
            "for merged report. \n"
            " Install with cars target : pdf_report \n"
            "pip install cars[pdf_report]"
        )


def generate_report_cars_output(report_file, output_dir, log_error, used_conf):
    """
    Generate a report PDF using HTML and WeasyPrint.

    :param report_file: path to the report pdf to generate
    :param output_dir: cars output directory
    :param log_error: error log string or status
    :param used_conf: configuration dictionary
    """
    # Configuration Extraction
    try:
        subsampling = used_conf.get("subsampling", {})
        advanced = subsampling.get("advanced", {})
        used_resolution = advanced.get("resolutions", [1])[0]
    except (KeyError, IndexError, AttributeError):
        used_resolution = 1

    conf_display_text = json.dumps(used_conf, indent=4)

    # Images generation
    images_dir = os.path.join(output_dir, "images")
    envelope_images = generate_envelope_images(
        output_dir, used_resolution, images_dir
    )
    epipolar_images = generate_epipolar_images(
        output_dir, used_resolution, images_dir
    )
    dsm_overview_images = generate_dsm_overview(output_dir, images_dir)

    sections_data = [
        ("Section 2: Envelopes", envelope_images),
        ("Section 3: Epipolar Images", epipolar_images),
        ("Section 4: DSM Overview", dsm_overview_images),
    ]

    # HTML & CSS Template
    styles = (
        "@page { size: A4; margin: 1.5cm; "
        "@bottom-right { content: 'Page ' counter(page); font-size: 9pt; } } "
        "body { font-family: sans-serif; line-height: 1.6; color: #2c3e50; } "
        "h1 { text-align: center; border-bottom: 3px solid #34495e; } "
        "h2 { color: #2980b9; border-bottom: 1px solid #eee; margin-top: 40px; "
        "page-break-before: always; } "
        "img { max-width: 100%; height: auto; display: block; "
        "margin: 10px auto; } "
        ".image-item { margin-bottom: 20px; text-align: center; "
        "page-break-inside: avoid; } "
        ".caption { font-size: 8pt; color: #7f8c8d; font-style: italic; } "
        ".config-container { background-color: #fdfdfe; "
        "border: 1px solid #dcdde1; "
        "padding: 15px; font-family: monospace; font-size: 8pt; "
        "white-space: pre-wrap; } "
        ".log-info { color: #c0392b; font-weight: bold; background: #f9ebea; "
        "padding: 10px; border-left: 5px solid #c0392b; }"
    )
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>{styles}</style>
    </head>
    <body>
        <h1>CARS Analysis Report</h1>
        <p style="text-align: center;">
            <strong>Resolution Level:</strong> {used_resolution}
        </p>

        <div class="section">
            <h2 style="page-break-before: avoid;">
                Section 1: Configuration & Logs
            </h2>
            <div class="log-info">Log Status: {log_error}</div>
            <p><strong>Configuration (used_conf):</strong></p>
            <div class="config-container">{conf_display_text}</div>
        </div>
    """

    #  Adding Image Sections Dynamically
    for title, images in sections_data:
        if images:
            html_content += f"<h2>{title}</h2><div class='image-grid'>"
            for img_path in images:
                if os.path.exists(img_path):
                    abs_path = os.path.abspath(img_path)
                    html_content += f"""
                    <div class="image-item">
                        <img src="file://{abs_path}">
                        <div class="caption">{os.path.basename(img_path)}</div>
                    </div>
                    """
            html_content += "</div>"

    html_content += "</body></html>"

    # Save html report
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    return report_file


def safe_list_return(func):
    """
    Wrapper to ensure that the decorated function returns a
    list even in case of exceptions.
    :param func:
    :return:
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """
        Wrapper to safely try to execute function
        :param args: args
        :param kwargs: kwargs
        :return:
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {e}")
            return []

    return wrapper


def normalize_image(img):
    """
    Normalize image for better visualization
    :param img: input image
    :return: clipped and normalized image
    """
    # On élimine les valeurs extrêmes (outliers) pour un meilleur contraste
    low, high = np.percentile(img, (2, 98))
    img_clipped = np.clip(img, low, high)
    return (img_clipped - low) / (high - low)


@safe_list_return
def generate_epipolar_images(output_dir, used_resolution, images_dir):
    """
    Generate the epipolar images overview

    :param output_dir: cars output directory
    :type output_dir: str
    :param used_resolution: used resolution for the report
    :type used_resolution: float
    :param images_dir: path to the directory to save the generated image
    :type images_dir: str

    :return path to the generated image
    :rtype: str
    """

    epipolar_images = []

    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    epipolar_image = os.path.join(images_dir, "epipolar_overview")

    base_dir = os.path.join(
        output_dir,
        "intermediate_data",
        "surface_modeling",
        "res" + str(used_resolution),
        "tie_points",
    )
    pairs = [
        direct
        for direct in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, direct))
    ]

    for pair in pairs:
        # Paths
        path_imgs = os.path.join(
            base_dir, pair, "dump_dir", "resampling", "initial", pair
        )
        path_matches = os.path.join(
            base_dir, pair, pair, "filtered_matches.npy"
        )

        img_l_path = os.path.join(path_imgs, "epi_img_left.tif")
        img_r_path = os.path.join(path_imgs, "epi_img_right.tif")

        if not (os.path.exists(img_l_path) and os.path.exists(path_matches)):
            logging.error("no matches or epipolar images found")
            continue

        with (
            rasterio.open(img_l_path) as src_l,
            rasterio.open(img_r_path) as src_r,
        ):
            img_l = normalize_image(src_l.read(1))
            img_r = normalize_image(src_r.read(1))

        matches = np.load(path_matches)

        for with_matches in [False, True]:
            fig, (ax1, ax2) = plt.subplots(
                1, 2, figsize=(22, 12), gridspec_kw={"wspace": 0.02}
            )

            ax1.imshow(img_l, cmap="gray")
            ax2.imshow(img_r, cmap="gray")

            # Title
            status = "with matches" if with_matches else " "
            fig.suptitle(
                f"Pair : {pair}  {status}",
                fontsize=20,
                fontweight="bold",
                y=0.95,
            )

            if with_matches:
                # Tracé des segments
                nb_draw = min(200, len(matches))
                indices = np.linspace(0, len(matches) - 1, nb_draw, dtype=int)

                for i in indices:
                    pt_l = (matches[i, 0], matches[i, 1])
                    pt_r = (matches[i, 2], matches[i, 3])

                    con = ConnectionPatch(
                        xyA=pt_r,
                        xyB=pt_l,
                        coordsA="data",
                        coordsB="data",
                        axesA=ax2,
                        axesB=ax1,
                        color=plt.cm.hsv(np.random.rand()),  # Couleurs variées
                        linewidth=0.8,
                        alpha=0.6,
                    )
                    ax2.add_artist(con)
                    ax1.plot(pt_l[0], pt_l[1], "r+", markersize=2, alpha=0.5)
                    ax2.plot(pt_r[0], pt_r[1], "r+", markersize=2, alpha=0.5)

            for ax in [ax1, ax2]:
                ax.set_axis_off()

            # Save
            suffix = "matches" if with_matches else "simple"
            output_name = f"{epipolar_image}_{pair}_{suffix}.png"
            plt.savefig(output_name, bbox_inches="tight", dpi=150)
            plt.close(fig)
            epipolar_images.append(output_name)

        # Generate matches analysis
        if len(matches) > 0:
            dx = matches[:, 3] - matches[:, 1]
            dy = matches[:, 2] - matches[:, 0]

            fig_disp, (ax_dx, ax_dy) = plt.subplots(1, 2, figsize=(16, 6))
            fig_disp.suptitle(
                f"Match Displacement Analysis - {pair}",
                fontsize=16,
                fontweight="bold",
                y=1.02,
            )

            #  Disparity
            ax_dx.hist(
                dy, bins=60, color="skyblue", edgecolor="black", alpha=0.7
            )
            ax_dx.axvline(
                np.mean(dy),
                color="red",
                linestyle="--",
                label=f"Mean : {np.mean(dy):.2f}",
            )
            ax_dx.set_title(" Column Disparity", fontsize=14)
            ax_dx.set_xlabel("Pixels")
            ax_dx.set_ylabel("Frequency")
            ax_dx.grid(axis="y", alpha=0.3)
            ax_dx.legend()

            # Epipolar Error
            ax_dy.hist(
                dx, bins=60, color="salmon", edgecolor="black", alpha=0.7
            )
            ax_dy.axvline(
                0,
                color="black",
                linestyle="-",
                linewidth=1.5,
                label="Ideal (0)",
            )
            ax_dy.axvline(
                np.mean(dx),
                color="red",
                linestyle="--",
                label=f"Mean: {np.mean(dx):.2f}",
            )
            ax_dy.set_title("Lign Disparity: Epipolar Error", fontsize=14)
            ax_dy.set_xlabel("Pixels")
            ax_dy.set_ylabel("Frequency")
            ax_dy.grid(axis="y", alpha=0.3)
            ax_dy.legend()

            plt.tight_layout()

            # Save the analysis plot
            analysis_filename = f"{epipolar_image}_{pair}_dx_dy_analysis.png"
            plt.savefig(analysis_filename, dpi=120, bbox_inches="tight")
            plt.close(fig_disp)
            epipolar_images.append(analysis_filename)

    return epipolar_images


@safe_list_return
def generate_envelope_images(output_dir, used_resolution, images_dir):
    """
    Generate the envelope interpretation image

    :param output_dir: cars output directory
    :type output_dir: str
    :param used_resolution: used resolution for the report
    :type used_resolution: float
    :param images_dir: path to the directory to save the generated image
    :type images_dir: str

    :return path to the generated image
    :rtype: str
    """
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    envelope_images = []

    # Generate envelope interpretation
    envelope_image = os.path.join(images_dir, "envelope_interpretation")

    metadata_file = os.path.join(
        output_dir,
        "intermediate_data",
        "surface_modeling",
        "res" + str(used_resolution),
        "metadata.yaml",
    )

    with open(metadata_file, "r", encoding="utf-8") as meta_file:
        metadata = yaml.safe_load(meta_file)
        pairs_envelopes = metadata["pair_preprocessing"]

    for pair_name, content in pairs_envelopes.items():
        # load geojson as GeoDataFrame
        gdf_left = gpd.GeoDataFrame.from_features(
            content["left_envelope"]["features"], crs="EPSG:4326"
        )
        gdf_right = gpd.GeoDataFrame.from_features(
            content["right_envelope"]["features"], crs="EPSG:4326"
        )

        _, ax = plt.subplots(figsize=(10, 10))

        # Plot Polygons
        gdf_left.plot(
            ax=ax,
            facecolor="none",
            edgecolor="blue",
            linewidth=2,
            label="Left Envelope",
        )
        gdf_right.plot(
            ax=ax,
            facecolor="none",
            edgecolor="red",
            linewidth=2,
            linestyle="--",
            label="Right Envelope",
        )

        #  Convert to 3857 for contextily
        gdf_left_3857 = gdf_left.to_crs(epsg=3857)
        gdf_right_3857 = gdf_right.to_crs(epsg=3857)

        # Show on 3857 for contextily
        ax.clear()
        gdf_left_3857.plot(
            ax=ax, facecolor="blue", alpha=0.3, edgecolor="blue", label="Left"
        )
        gdf_right_3857.plot(
            ax=ax, facecolor="red", alpha=0.3, edgecolor="red", label="Right"
        )

        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

        plt.title(f"Pair envelope : {pair_name}")
        ax.set_axis_off()

        # Save
        pair_image = envelope_image + "_" + pair_name + ".png"
        envelope_images.append(pair_image)
        plt.savefig(pair_image, bbox_inches="tight", dpi=150)
        plt.close()

    return envelope_images


@safe_list_return
def generate_dsm_overview(output_dir, images_dir):
    """
    Generate the dsm overview image

    :param output_dir: cars output directory
    :type output_dir: str
    :param images_dir: path to the directory to save the generated image
    :type images_dir: str

    :return path to the generated image
    :rtype: str
    """

    overview_images = []

    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    dsm_overview_image = os.path.join(images_dir, "dsm_overview.png")
    color_overview_image = os.path.join(images_dir, "color_dsm_overview.png")
    hillshade_overview_image = os.path.join(
        images_dir, "hillshade_dsm_overview.png"
    )

    # Load DSM
    dsm_path = os.path.join(output_dir, "dsm", "dsm.tif")
    hillshade_path = os.path.join(output_dir, "dsm", "hillshade.tif")
    color_path = os.path.join(output_dir, "dsm", "image.tif")
    # Generate hillsade
    generate_hillshade(dsm_path, hillshade_path)

    # Generate overviews
    save_robust_overview(
        dsm_path, dsm_overview_image, title="DSM Overview", cmap="gray"
    )

    save_robust_overview(
        color_path, color_overview_image, title="Image Overview", cmap="gray"
    )

    save_robust_overview(
        hillshade_path,
        hillshade_overview_image,
        title="DSM Overview",
        cmap="gray",
    )

    overview_images.append(dsm_overview_image)
    overview_images.append(color_overview_image)
    overview_images.append(hillshade_overview_image)

    return overview_images


def save_robust_overview(input_path, output_png, title, cmap=None):
    """
    Convert to overview
    :param input_path:
    :param output_png:
    :param title:
    :param cmap:
    :return:
    """

    with rasterio.open(input_path) as src:
        # Lecture et gestion des dimensions
        if src.count == 1:
            data = src.read(1)
            nodata = src.nodata
            mask = (data == nodata) if nodata is not None else np.isnan(data)

            valid_data = data[~mask]
            vmin, vmax = (
                np.percentile(valid_data, (2, 98))
                if valid_data.size > 0
                else (0, 1)
            )
            display_data = data
        else:
            # Multiband
            data = src.read([1, 2, 3])
            data = np.transpose(data, (1, 2, 0))

            # Per band normalization
            data_norm = data.astype(float)
            for i in range(3):
                low, high = np.percentile(data_norm[:, :, i], (2, 98))
                if high > low:
                    data_norm[:, :, i] = np.clip(
                        (data_norm[:, :, i] - low) / (high - low), 0, 1
                    )
            display_data = data_norm
            vmin, vmax = None, None

        # Figure generation
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(display_data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
        ax.set_axis_off()
        plt.savefig(output_png, bbox_inches="tight", dpi=150, facecolor="white")
        plt.close(fig)


def generate_hillshade(dsm_path, output_path, azimuth=315, altitude=45):
    """
    Generates a hillshade from a Digital Surface Model (DSM).

    Parameters:
    - dsm_path: Path to the input .tif file
    - output_path: Path to save the generated hillshade
    - azimuth: Direction of the light source (0-360, 315 is North-West)
    - altitude: Solar elevation angle (0-90)
    """
    with rasterio.open(dsm_path) as src:
        dsm = src.read(1).astype(np.float32)
        res_x, res_y = src.res
        meta = src.meta.copy()

    # Calculate Gradients (Slopes)
    dx, dy = np.gradient(dsm, res_x, res_y)

    #  Convert lighting parameters to radians
    # We use 360 - azimuth to convert from compass heading to mathematical angle
    azimuth_rad = np.radians(360 - azimuth)

    # Compute Slope and Aspect
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dy, dx)

    # Compute Hillshade
    # Formula: cos(Zenith) * cos(Slope) +
    # sin(Zenith) * sin(Slope) * cos(Azimuth - Aspect)
    zenith_rad = np.radians(90 - altitude)

    hillshade = np.cos(zenith_rad) * np.cos(slope) + np.sin(
        zenith_rad
    ) * np.sin(slope) * np.cos(azimuth_rad - aspect)

    # Scale result to 0-255 range (8-bit)
    hillshade = (hillshade * 255).clip(0, 255).astype(np.uint8)

    # Save the output
    meta.update(dtype=rasterio.uint8, count=1, driver="GTiff", nodata=None)
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(hillshade, 1)


def generate_satellite_plot_mpl(sat_positions, output_png):
    """
    Generate a satellite position plot using Matplotlib.
    :param sat_positions: list of satelite positions
    :param output_png: path to save the generated image
    :return:
    """
    plt.rc("grid", color="#316931", linewidth=1, linestyle="-")
    plt.rc("xtick", labelsize=15)
    plt.rc("ytick", labelsize=15)

    # force square figure and square axes looks better for polar, IMO
    width, height = plt.rcParams["figure.figsize"]
    size = min(width, height)
    # make a square figure
    fig = plt.figure(figsize=(size, size))

    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    for sat_prn, sat_az, sat_e in sat_positions:
        ax.annotate(
            str(sat_prn),
            xy=(radians(sat_az), 90 - sat_e),  # theta, radius
            bbox={"boxstyle": "round", "fc": "green", "alpha": 0.5},
            horizontalalignment="center",
            verticalalignment="center",
        )

    ax.set_yticks(range(0, 90 + 10, 10))
    ylabel = ["90", "", "", "60", "", "", "30", "", "", ""]
    ax.set_yticklabels(ylabel)
    plt.savefig(output_png, bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)


def generate_satelite_position_report(sat_positions, sat_infos, output_dir):
    """
    Geneate a report with the satellite positions and infos
    :param sat_positions:
    :param sat_infos:
    :param output_dir:
    :return:
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    images_dir = os.path.join(output_dir, "images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    image_path = os.path.join(images_dir, "satellite_positions.png")
    image_path = os.path.abspath(image_path)

    generate_satellite_plot_mpl(sat_positions, image_path)

    # Generate pdf
    if isinstance(sat_infos, dict):
        clean_infos = json.loads(
            json.dumps(
                sat_infos,
                default=lambda x: x.item() if hasattr(x, "item") else str(x),
            )
        )
        yaml_content = yaml.dump(clean_infos, sort_keys=False)
        infos_html = f"<pre class='yaml-box'>{yaml_content}</pre>"
    else:
        infos_html = f"<p>{sat_infos}</p>"

    style_css = """
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 40px;
            color: #333;
        }
        h1 {
            color: #1a5f7a;
            border-bottom: 2px solid #1a5f7a;
            padding-bottom: 10px;
        }
        .info-box {
            background: #f4f7f6;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        ul { list-style-type: none; padding: 0; }
        li { margin-bottom: 5px; }
        img { max-width: 100%; height: auto; border: 1px solid #ccc; }
    </style>
    """

    html_content = f"""
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        {style_css}
    </head>
    <body>
        <h1>Satellite Images</h1>
        <div class="info-box">
            {infos_html}
        </div>
        <img src="file://{image_path}">
    </body>
    </html>
    """

    report_file = os.path.join(output_dir, "report.html")

    # Save html report
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    return report_file
