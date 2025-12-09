.. _subsampling:

Subsampling
===========

The subsampling pipeline takes sensor images as inputs and generates subsampled versions at the resolutions specified in the configuration.
A sensor image may be accompanied by classifications, which will also be downsampled.
The list of resolutions this pipeline outputs is the list that will be used in the meta pipeline.

This pipeline does not run at multiple resolutions.

Allowed inputs
--------------

Subsampling can take only sensor images as inputs (and their respective classification), as shown in the :ref:`input <input>` section of the documentation.

Applications
------------

**Name**: "sensors_subsampling"

**Description**

Subsample the sensor images.

**Configuration**

+---------------------------------+--------------------------------------------------------------------------+------------+--------------------------------------+------------------------+----------+
| Name                            | Description                                                              | Type       | Available value                      | Default value          | Required |
+=================================+==========================================================================+============+======================================+========================+==========+
| method                          | Method for sensors_subsampling                                           | string     | "rasterio"                           | "rasterio"             | No       |
+---------------------------------+--------------------------------------------------------------------------+------------+--------------------------------------+------------------------+----------+
| tile_size                       | Size of tiling                                                           | int        |                                      | 10000                  | No       |
+---------------------------------+--------------------------------------------------------------------------+------------+--------------------------------------+------------------------+----------+
| interpolator_image              | Interpolator that should be used for the image                           | str        |                                      | "bilinear"             | No       |
+---------------------------------+--------------------------------------------------------------------------+------------+--------------------------------------+------------------------+----------+
| interpolator_classif            | Interpolator that should be used for the classif                         | str        |                                      | "nearest"              | No       |
+---------------------------------+--------------------------------------------------------------------------+------------+--------------------------------------+------------------------+----------+
| interpolator_mask               | Interpolator that should be used for the mask                            | str        |                                      | "nearest"              | No       |
+---------------------------------+--------------------------------------------------------------------------+------------+--------------------------------------+------------------------+----------+
| overlap                         | Overlap for margins                                                      | int        | should be > 0                        | 2                      | No       |
+---------------------------------+--------------------------------------------------------------------------+------------+--------------------------------------+------------------------+----------+

Advanced Parameters
-------------------

.. list-table::
    :header-rows: 1

    * - Name
      - Description
      - Type
      - Default value
    * - save_intermediate_data
      - Save intermediate data for all applications inside this pipeline.
      - bool
      - False
    * - resolutions
      - The resolution(s) at which the outputs should be resampled.
      - list[int]
      - [16, 4, 1]

Below is an example configuration for this pipeline :

.. include-cars-config:: ../../example_configs/pipeline/subsampling_pipeline
