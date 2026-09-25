.. _oversampling_app:

Oversampling
============

**Name**: "oversampling"

**Description**

Upsamples the outputs of :ref:`depth_map_generation <depth_map_generation_app>`
from the monocular working resolution to the original sensor image resolution.

This application is used in the monocular pipeline to produce the final edge map
at sensor resolution.

.. warning::

  This application is only available if the `CARS Monocular plugin <https://github.com/CNES/cars-monocular-plugin>`_ is installed.

**Configuration**

+------------------------+--------------------------------------------------------+---------+-----------------+---------------+----------+
| Name                   | Description                                            | Type    | Available value | Default value | Required |
+========================+========================================================+=========+=================+===============+==========+
| method                 | Oversampling method                                    | string  | "simple"        | "simple"      | No       |
+------------------------+--------------------------------------------------------+---------+-----------------+---------------+----------+
| save_intermediate_data | Whether to save intermediate data                      | boolean | true/false      | false         | No       |
+------------------------+--------------------------------------------------------+---------+-----------------+---------------+----------+

Method
------

The available oversampling implementation is:

- ``simple``: rescales the tiled monocular outputs to the original sensor size.

Outputs
-------

This application always saves:

- ``edges.tif``: oversampled edge map at the original sensor resolution.

When intermediate data saving is enabled, it also saves:

- ``depth.tif``: oversampled depth map.
- ``normals.tif``: oversampled normal map.
- ``tile_id.tif``: oversampled tile identifier map.

**Example**

.. include-cars-config:: ../../example_configs/configuration/applications_oversampling