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

| Example files are available on AWS S3 through the SpaceNet challenge here: s3://spacenet-dataset/Hosted-Datasets/MVS_dataset/WV3/PAN/.
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

.. _`OTB`: https://www.orfeo-toolbox.org/CookBook-8.0/C++/UserGuide.html#image-data-representation

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

To translate single image or multiband image with several nbits per band to 1bit per band, it can be recommended to use gdal_translate as follows:

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

CARS generates several laz files corresponding to the tiles processed.
Merge can be done with `laszip`_. 

To merge them:

.. code-block:: console

    laszip -i data\*.laz -merged -o merged.laz


.. _`laszip`: https://laszip.org/
