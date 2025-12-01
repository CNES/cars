.. _resampling_app:

Resampling
==========

**Name**: "resampling"

**Description**

Input images are resampled with grids.

**Configuration**

+------------------------+--------------------------------------------------------+---------+-----------------+---------------+----------+
| Name                   | Description                                            | Type    | Available value | Default value | Required |
+========================+========================================================+=========+=================+===============+==========+
| method                 | Method for resampling                                  | string  | "bicubic"       | "bicubic"     | No       |
+------------------------+--------------------------------------------------------+---------+-----------------+---------------+----------+
| strip_height           | Height of strip (only when tiling is done by strip)    | int     | should be > 0   | 60            | No       |
+------------------------+--------------------------------------------------------+---------+-----------------+---------------+----------+
| step                   | Horizontal step for resampling inside a strip          | int     | should be > 0   | 500           | No       |
+------------------------+--------------------------------------------------------+---------+-----------------+---------------+----------+
| save_intermediate_data | Save epipolar images and texture                       | boolean |                 | false         | No       |
+------------------------+--------------------------------------------------------+---------+-----------------+---------------+----------+

**Example**

.. include-cars-config:: ../../example_configs/configuration/applications_resampling