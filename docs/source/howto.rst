.. _howto:

=========
How to...
=========

.. _get_stereo_products:

Get full stereo products
========================


Dinamis
-------

| DINAMIS is a platform that acquires and distributes satellite Earth imagery for French and foreign institutional users under `specific subscription conditions <https://dinamis.data-terra.org/en/eligible-users/>`_.
| Please visit the dinamis website for more information: https://dinamis.data-terra.org/.


AIRBUS Pleiades NEO example files
---------------------------------
Example files are available here: https://intelligence.airbus.com/imagery/sample-imagery/pleiades-neo-tristereo-marseille/ (A form must be filled out to access the data).

.. _maxar_example_files:

Maxar WorldView example files
-----------------------------

| Example files are available on AWS S3 through the SpaceNet challenge here: `s3://spacenet-dataset/Hosted-Datasets/MVS_dataset/WV3/PAN/`
| You need to install `aws-cli <https://github.com/aws/aws-cli>`_:

.. code-block:: console

   python -m venv venv-aws-cli # create a virtual environment
   source ./venv-aws-cli/bin/activate # activate it
   pip install --upgrade pip # upgrade pip
   pip install awscli


And download a stereo:

.. code-block:: console

   aws s3 cp --no-sign-request s3://spacenet-dataset/Hosted-Datasets/MVS_dataset/WV3/PAN/18DEC15WV031000015DEC18140522-P1BS-500515572020_01_P001_________AAE_0AAAAABPABJ0.NTF .
   aws s3 cp --no-sign-request s3://spacenet-dataset/Hosted-Datasets/MVS_dataset/WV3/PAN/18DEC15WV031000015DEC18140554-P1BS-500515572030_01_P001_________AAE_0AAAAABPABJ0.NTF  .


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

		

For example, if you have downloaded the maxar example data :ref:`maxar_example_files`, you can choose a region of interest with `geojson.io <https://geojson.io/#map=16.43/-34.490433/-58.586864>`_.

And then extract region, create config file and launch cars:

.. code-block:: console

   cars-extractroi -il *.NTF -out ext_dir -bbx -58.5896 -34.4872 -58.5818 -34.4943
   cars-starter -il ext_dir/*.tif -out out_dir > config.json
   cars config.json


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

    otbcli_BundleToPerfectSensor -inp image.tif -inxs color.tif -out color_pxs.tif

.. code-block:: console

    docker run -w /data -v "$(pwd)"/data:/data --entrypoint=/bin/bash  cnes/cars otbcli_BundleToPerfectSensor -inp /data/image.tif -inxs /data/color.tif -out /data/color_pxs.tif

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

See next section to apply a gdal_translate to convert the mask with 1bit image struture.

.. _convert_image_to_binary_image:

Convert image to binary image
-----------------------------

To translate single image or multiband image with several nbits per band to 1bit per band, it can be recommended to use `gdal_translate <https://gdal.org/en/latest/programs/gdal_translate.html>`_ as follows:

.. code-block:: console

    gdal_translate -ot Byte -co NBITS=1 mask.tif mask_1nbit.tif

.. _add_band_description_in_image:

Add band name / description in tiff files metadata
--------------------------------------------------

To add a band name /description in tiff files, for classification or color files in order to be used:


.. code-block:: python

    data_in = gdal.Open(infile, gdal.GA_Update)
    band_in = data_in.GetRasterBand(inband)
    band_in.SetDescription(band_description)
    data_in = None


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

    docker run -w /data -v "$(pwd)"/data_gizeh_small:/data cnes/cars /data/configfile.json

Use OTB in docker
-----------------

Any OTB application can be ran in docker

.. code-block:: console

    docker run  --entrypoint=/bin/bash  cnes/cars otbcli_BandMath -help


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



Use CARS with Pleiades images ...
========================================


.. _pleiade_raw_data:

... with raw data
-----------------


If you want to generate a 3D model with the following pair:

.. code-block:: bash

    IMG_PHR1B_MS_003
    IMG_PHR1B_MS_004
    IMG_PHR1B_P_001
    IMG_PHR1B_P_002

You should find in each folder the following data:

.. code-block:: bash

    ...
    DIM_PHR1B_***.XML
    IMG_PHR1B_***.TIF
    RPC_PHR1B_***.XML


For each product, the user must provide the path to the pancromatic data (*P*.TIF) with its geomodel, all contained in the DIMAP file (DIMAP*P*.XML):


.. code-block:: json

    {
    "inputs": {
        "sensors" : {
            "one": {
                "image": "IMG_PHR1B_P_001/DIM_PHR1B_***.XML"
            },
            "two": {
                "image": "IMG_PHR1B_P_002/DIM_PHR1B_***.XML",
            }
        },
        "pairing": [["one", "two"]]
        }
    }



If you want to add the colors, a P+XS fusion must be done, to specify a color.tif with the same shape and resolution than the Pancromatic data.
It can be performed with `otbcli_BundleToPerfectSensor` as explained in  `make_a_simple_pan_sharpening`_

.. code-block:: json

    {
    "inputs": {
        "sensors" : {
            "one": {
                "image": "IMG_PHR1B_P_001/DIM_PHR1B_***.XML",
                "color": "color_one.tif"
            },
            "two": {
                "image": "IMG_PHR1B_P_002/DIM_PHR1B_***.XML",
                "color": "color_two.tif"

            }
        },
        "pairing": [["one", "two"]]
        }
    }




.. _pleiade_roi_data:

... with a region of interest
-----------------------------

There are two different uses of roi in CARS:

* Crop input images: the whole pipeline will be done with cropped images
* Use input roi parameter: the whole images will be used to compute grid correction and terrain + epipolar a priori. Then the rest of the pipeline will use the given roi. This allow better correction of epipolar rectification grids.


If you want to only work with a region of interest for the whole pipeline, use cars-extractroi:

.. code-block:: bash

    cars-extractroi -il DIM_PHR1B_***.XML -out ext_dir -bbx -58.5896 -34.4872 -58.5818 -34.4943

It generates a .tif and .geom to be used as:

.. code-block:: json

    {
    "inputs": {
        "sensors" : {
            "one": {
                "image": "ext_dir/***.tif",
                "geomodel": "ext_dir/***.geom",
                "color": "color_one.tif"
            }
    }

And use generated data as previously explained with raw data.


If you want to compute grid correction and compute epipolar/ terrain a priori on the whole image, keep the same input images, but specify terrain ROI to use:

.. code-block:: json

    {
        "inputs":
        {
            "roi" : {
                "type": "FeatureCollection",
                "features": [
                    {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "coordinates": [
                        [
                            [5.194, 44.2064],
                            [5.194, 44.2059],
                            [5.195, 44.2059],
                            [5.195, 44.2064],
                            [5.194, 44.2064]
                        ]
                        ],
                        "type": "Polygon"
                    }
                    }
                ]
            }
        }
    }



See  Usage Sensors Images Inputs configuration for more information.
