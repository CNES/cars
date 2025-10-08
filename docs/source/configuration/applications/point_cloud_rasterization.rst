Point Cloud Rasterization
=========================

**Name**: "point_cloud_rasterization"

**Description**

Project altitudes on regular grid.

Only one simple gaussian method is available for now.

.. list-table:: Configuration
    :widths: 19 19 19 19 19 19
    :header-rows: 1

    * - Name
      - Description
      - Type
      - Available value
      - Default value
      - Required
    * - method
      -
      - string
      - "simple_gaussian"
      - simple_gaussian
      - No
    * - dsm_radius
      -
      - float, int
      -
      - 1.0
      - No
    * - sigma
      -
      - float
      -
      - None
      - No
    * - grid_points_division_factor
      -
      - int
      -
      - None
      - No
    * - dsm_no_data
      -
      - int
      -
      - -32768
      -
    * - texture_no_data
      - If texture_no_data is None, it will be automatically set to the maximum value of texture_dtype
      - int, None
      -
      - None
      -
    * - texture_dtype
      - By default, it's retrieved from the input texture. Otherwise, specify an image type
      - string
      - "uint8", "uint16", "float32" ...
      - None
      - No
    * - msk_no_data
      - No data value for mask  and classif
      - int
      -
      - 255
      -
    * - save_intermediate_data
      - Save all layers from input point cloud in application `dump_dir`
      - boolean
      -
      - false
      - No

**Example**

.. include-cars-config:: ../../example_configs/configuration/applications_point_cloud_rasterization
