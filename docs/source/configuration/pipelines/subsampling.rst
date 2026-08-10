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

The subsampling pipeline uses these applications:

  - :ref:`sensors_subsampling <sensors_subsampling_app>`


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
    * - min_image_size
      - Image size below which subsampling is not done.
      - int
      - 80

Below is an example configuration for this pipeline :

.. include-cars-config:: ../../example_configs/pipeline/subsampling_pipeline
