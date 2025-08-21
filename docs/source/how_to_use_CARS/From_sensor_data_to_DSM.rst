====================================================
From sensor data to DSM : the step-by-step tutorials
====================================================

Use CARS with Pleiades images ...
========================================

.. note::
	CARS is used in the same way with Pléiades and Pléiades Neo data.

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

.. include-cars-config:: ../example_configs/how_to_use_CARS/from_sensor_data_to_DSM/dimap_1

If you want to add the colors, a P+XS fusion must be done, to specify a color.tif with the same shape and resolution than the Pancromatic data.
It can be performed with `otbcli_BundleToPerfectSensor` as explained in  :ref:`make_a_simple_pan_sharpening`.

.. include-cars-config:: ../example_configs/how_to_use_CARS/from_sensor_data_to_DSM/dimap_2

.. _pleiade_roi_data:

... with a region of interest
-----------------------------

There are two different ways to use a ROI in CARS:

* Crop input images: the whole pipeline will be done with cropped images
* Use input roi parameter: the whole images will be used to compute grid correction and terrain + epipolar a priori. Then the rest of the pipeline will use the given roi. This allows a better correction of epipolar rectification grids.


If you want to work with cropped image by using a region of interest for the whole pipeline, use cars-extractroi:

.. code-block:: bash

    cars-extractroi -il DIM_PHR1B_***.XML -out ext_dir -bbx -58.5896 -34.4872 -58.5818 -34.4943

It generates a .tif and .geom to be used as:

.. include-cars-config:: ../example_configs/how_to_use_CARS/from_sensor_data_to_DSM/with_a_roi_1

Then use the generated data as you would with raw data.


If you want to compute the grid correction and compute the epipolar/terrain a priori on the whole image, keep the same input images but specify the terrain ROI to use:

.. include-cars-config:: ../example_configs/how_to_use_CARS/from_sensor_data_to_DSM/with_a_roi_2

See  Usage Sensors Images Inputs configuration for more information.

.. note::
	CARS also works with other types of data: SPOT 6-7, WorldView, etc.
