.. _howto:

===============
Additional tips
===============

Prepare input images
====================

.. _make_input_roi_images:

Make input ROI images
---------------------

``cars-extractroi`` script allows to extract region of interest from your image product.

.. code-block:: console

   usage: cars-extractroi [-h] -il [IL [IL ...]] -out OUT -bbx x1 y1 x2 y2

   Helper to extract roi from bounding box

   optional arguments:
     -h, --help         show this help message and exit
     -il [IL [IL ...]]  Image products
     -out OUT           Extracts directory
     -bbx x1 y1 x2 y2   Bounding box from two points (x1, y1) and (x2, y2)

		
        
How to find the coordinates of the bounding box ?
.................................................

For example, if you have downloaded the maxar example data :ref:`maxar_example_files`, you are working in an area near to San Fernando in Argentina. Go to the website `geojson.io <https://geojson.io/>`_ in order to select your ROI:

.. |roisanfernando| image:: ../images/roi_san_fernando_argentina.jpg
   :width: 60%

|roisanfernando|

You can either select the upper left corner with the lower right corner (in red in the previous image): 

.. code-block:: console

   cars-extractroi -il *.NTF -out ext_dir -bbx -58.5809 -34.4934 -58.5942 -34.4869
   cars-starter -il ext_dir/*.tif -out out_dir > config.yaml
   cars config.yaml

or the lower left corner with the upper right corner (in purple in the previous image):

.. code-block:: console

   cars-extractroi -il *.NTF -out ext_dir -bbx -58.5809 -34.4869 -58.5942 -34.4934
   cars-starter -il ext_dir/*.tif -out out_dir > config.yaml
   cars config.yaml


N.B.: Instead of using ``cars-extractroi``, you can directly give the GeoJson dictionnary in the configuration file (Please, see :ref:`input` for details). In this case, the sparse steps (geometric corrections) are processed on the entire image and not only on the ROI.

Monitor tiles progression
-------------------------

``cars-dashboard`` script allows to monitor the progression of tiles computation on a web browser.

.. code-block:: console

    usage: cars-dashboard [-h] -out OUT

    Helper to monitor tiles progress

    optional arguments:
    -h, --help  show this help message and exit
    -out OUT    CARS output folder to monitor

For example, if you want to monitor the computation of a CARS run:

.. code-block:: console

    cars-dashboard -out output_cars


.. _make_a_simple_pan_sharpening:


Make a simple pan sharpening
----------------------------

In the case of Pleiades sensors, the XS color isn't superimposable to the Panchromatic image.

It can be recommended to apply a P+XS pansharpening with `OTB`_.

.. code-block:: console

    otbcli_BundleToPerfectSensor -method bayes -inp image.tif -inxs color.tif -out color_pxs.tif

.. code-block:: console

    docker run -w /data -v "$(pwd)"/data:/data --entrypoint=/bin/bash  cnes/cars otbcli_BundleToPerfectSensor -method bayes -inp /data/image.tif -inxs /data/color.tif -out /data/color_pxs.tif

.. _`OTB`: https://www.orfeo-toolbox.org/CookBook-8.0/C++/UserGuide.html#image-data-representation

Convert RGB image to panchromatic image
---------------------------------------

CARS only uses panchromatic images for processing.

If you have a multi-spectral image, you'll need to extract a single band to use, or convert it to a panchromatic image before using it with CARS.

The line below use `"Grayscale Using Luminance" <https://en.wikipedia.org/wiki/Grayscale#Luma_coding_in_video_systems>`_ expression with `OTB BandMath <https://www.orfeo-toolbox.org/CookBook/Applications/app_BandMath.html>`_


.. code-block:: console

    otbcli_BandMath -il image.tif -out image_panchromatic.tif -exp "(0.2126 * im1b1 + 0.7152 * im1b2 + 0.0722 * im1b3)"


.. _make_a_water_mask:

Make a water mask
-----------------

To produce a water mask from R,G,B,NIR images, it can be recommended to compute a Normalized Difference Water Index (NDWI) and threshold the output to a low value.

The low NDWI values can be considered as water area.

.. code-block:: console

    gdal_calc.py -G input.tif --G_band=2 -N input.tif --N_band=4 --outfile=mask.tif --calc="((1.0*G-1.0*N)/(1.0*G+1.0*N))>0.3" --NoDataValue=0
    
It is also possible to produce a water mask with `SLURP <https://github.com/CNES/slurp>`_.   

See next section to apply a gdal_translate to convert the mask with 1bit image struture.

.. _convert_image_to_binary_image:

Convert image to binary image
-----------------------------

To translate single image or multiband image with several nbits per band to 1bit per band, it can be recommended to use `gdal_translate <https://gdal.org/en/latest/programs/gdal_translate.html>`_ as follows:

.. code-block:: console

    gdal_translate -ot Byte -co NBITS=1 mask.tif mask_1nbit.tif


.. _download_srtm_tiles:

Get low resolution DEM
========================

SRTM 90m DEM
---------------

It is possible to download a low resolution DEM (90-m SRTM) corresponding to your area. To get a SRTM tile, you need to run the following python script knowing the latitude and the longitude of your area:

.. code-block:: python

    import numpy as np

    def get_srtm_tif_name(lat, lon):
        """Download srtm tiles"""
        # longitude: [1, 72] == [-180, +180]
        tlon = (1+np.floor((lon+180)/5)) % 72
        tlon = 72 if tlon == 0 else tlon

        # latitude: [1, 24] == [60, -60]
        tlat = 1+np.floor((60-lat)/5)
        tlat = 24 if tlat == 25 else tlat

        srtm = "https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/srtm_%02d_%02d.zip" % (tlon, tlat)
        return srtm

    if __name__ == "__main__":
        print("Get SRTM tile corresponding to latitude and longitude couple")
        while 1:
            print(">> Latitude? ", end="")
            lat = input()
            print(">> Longitude? ", end="")
            lon = input()
            print(">> SRTM filename:", get_srtm_tif_name(int(lat), int(lon)))
            input()


If your area intersects multiple latitudes and longitudes, get all the SRTM tiles and create a VRT from them:

.. code-block:: console

    gdalbuildvrt srtm.vrt srtm_tile1.tif srtm_tile2.tif

Post process output
===================

.. _merge_laz_files:

Merge Laz files
---------------

CARS generates several `laz files <https://docs.fileformat.com/gis/laz/>`_ corresponding to the tiles processed.

To merge them:

.. code-block:: console

    laszip -i data\*.laz -merged -o merged.laz


.. _`laszip`: https://laszip.org/


Docker
======

A docker is available to use CARS and OTB applications.
CARS is the docker entrypoint. To use otb, entrypoint must be specified.

Use CARS in docker
------------------

.. code-block:: console

    docker run -w /data -v "$(pwd)"/data_gizeh_small:/data cnes/cars /data/configfile.yaml

Use OTB in docker
-----------------

Any OTB application can be ran in docker

.. code-block:: console

    docker run  --entrypoint=/bin/bash  cnes/cars otbcli_BandMath -help
    
You can either enter docker's interactive mode or execute the program from outside of the docker, as explained right below (example for extract-roi): 

Interactive mode:
-----------------

You can enter in the docker interactive mode by using this command : 

.. code-block:: console

    docker run -it -w /data -v "$(pwd)"/data_gizeh_small:/data  --entrypoint /bin/bash cnes/cars:latest

+--------------------------------------+------------------------------------------------------------------------------+
| Option                               | Explication                                                                  |
+======================================+==============================================================================+
| *docker run*                         | Runs a container based on the cnes/cars:latest image.                        |
+--------------------------------------+------------------------------------------------------------------------------+
| *-it*                                | Interactive mode (-i: interactive input, -t: allocates a pseudo-terminal).   |
+--------------------------------------+------------------------------------------------------------------------------+
| *-w /data*                           | Sets /data as the working directory inside the container.                    |
+--------------------------------------+------------------------------------------------------------------------------+
| *-v "$(pwd)"/data_gizeh_small:/data* | Mounts the local data_gizeh_small folder to /data inside the container.      |
+--------------------------------------+------------------------------------------------------------------------------+
| *--entrypoint /bin/bash*             | Overrides the container’s default entrypoint to run /bin/bash instead.       |
+--------------------------------------+------------------------------------------------------------------------------+
| *cnes/cars:latest*                   | Uses the cnes/cars:latest Docker image containing cars-extractroi.           |
+--------------------------------------+------------------------------------------------------------------------------+

From outside
------------

The other option is to directly use this complete command:

.. code-block:: console

    docker run  -w /data -v "$(pwd)"/data_gizeh_small:/data  --entrypoint cars-extractroi cnes/cars:latest -il /data/img1.tif -out crop_img1.tif  -bbx 20800 5100 21000 5300

.. _resample_image:

Resample an image
========================

If you want to upscale or downscale the resolution of you input data, use rasterio:

.. code-block:: python

    import rasterio
    from rasterio.enums import Resampling
    # Get data
    upscale_factor = 0.5
    with rasterio.open("example.tif") as dataset:
        # resample data to target shape
        data = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height * upscale_factor),
                int(dataset.width * upscale_factor)
            ),
            resampling=Resampling.bilinear
        )
        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]),
            (dataset.height / data.shape[-2])
        )
        profile = dataset.profile
        # Save data
        profile.update(
            width=data.shape[-1],
            height=data.shape[-2],
            transform=transform
        )
        with rasterio.open('resampled_example.tif', 'w', **profile) as dst:
            dst.write(data)


.. _ground_truth_reprojection_tutorial:

How to use the ground truth reprojection application
=======================================================

The ground truth reprojection application allows you to reproject a reference DSM and its associated auxiliary data into sensor or epipolar geometry.

To enable this application, select the desired reprojection target in the `surface_modeling` configuration:

.. code-block:: yaml

	surface_modeling:
	  applications:
	    all:
	      ground_truth_reprojection:
	        method: direct_loc
	        target: all

The `target` parameter specifies the geometry in which the reference data will be generated:

* `epipolar`: generates reference disparity values in epipolar geometry.
* `sensor`: generates reference elevation values in sensor geometry.
* `all`: generates both representations.

Configuring the Reference Data
--------------------------------

The reference DSM and its associated auxiliary data must be specified in the advanced configuration section of `surface_modeling`:

.. code-block:: yaml

	surface_modeling:
	  advanced:
	    all:
	      ground_truth_dsm:
	        dsm: path/to/ground_truth_dsm.tif
	        auxiliary_data:
	          classification: path/to/classification.tif
	          texture: path/to/texture.tif
	        auxiliary_data_interpolation:
	          classification: nearest
	          texture: linear

The configuration includes the following parameters:

* `dsm`: path to the reference DSM used to provide ground-truth elevation information.
* `auxiliary_data`: paths to auxiliary datasets associated with the reference DSM, such as classification maps or texture images.
* `auxiliary_data_interpolation`: interpolation methods used to resample each auxiliary dataset into the target geometry.

The interpolation method should be selected according to the data type. For example, `nearest` is generally suitable for categorical data, such as classification maps, while `linear` can be used for continuous data.

Reprojection Modes
-------------------

When `target` is set to `all`, the application generates reference data in both sensor and epipolar geometry.

**Epipolar geometry**

Positions on the epipolar grid are converted into physical coordinates and then mapped to pixel coordinates in the sensor image. The reference geometric model is used to compute ground-truth altitudes, which are compared with the corresponding altitudes obtained from the geometric model associated with the DSM generated from the input images.

The altitude difference is converted into a reference disparity using the `disp_to_alt_ratio` factor.

**Sensor geometry**

Pixel positions in the sensor image are geolocated using the ground-truth geometric model. The resulting altitudes form the reference elevation map.

Auxiliary Data Resampling
---------------------------

Auxiliary datasets are resampled into the target geometry using the geographic coordinates obtained through the direct localization of the reference DSM. These coordinates are used to locate the corresponding pixels in each auxiliary dataset and interpolate their values onto the target grid.

The resampled auxiliary data are aligned with the target geometry, allowing information such as terrain or land-cover classifications to be associated with the same locations as the reference data.

Output
--------

The application produces ground-truth data in the requested geometry:

* In **sensor geometry**, the output contains reference elevation values.
* In **epipolar geometry**, the output contains reference disparity values.
* For each configured auxiliary dataset, the corresponding values are resampled into the target geometry using the specified interpolation method.

These outputs provide a geometric reference for evaluating the DSM generated from the input images and allow auxiliary information to be used consistently in subsequent processing steps.

