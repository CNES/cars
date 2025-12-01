.. _edge_detection:

Edge Detection
==============

This pipeline produces a regularization mask from an input image. The regularizaton mask is typically used by ``surface_modeling``, to guide stereo matching in areas with a high density of discontinuities, such as cities.

Allowed inputs
--------------

Edge detection can take only sensor images as inputs, as shown in the :ref:`input <input>` section of the documentation.

Applications
------------

**WIP** : the application(s) in edge detection have not been created yet.

Advanced Parameters
-------------------

.. list-table:: Configuration
    :header-rows: 1

    * - Name
      - Description
      - Type
      - Default value
    * - save_intermediate_data
      - Save intermediate data for all applications inside this pipeline. See :ref:`save_intermediate_data <save_intermediate_data>`
      - bool
      - False
