.. _resampling_app:

Resampling
==========

**Name**: "resampling"

**Description**

Input images are resampled with grids.

**Configuration**

+------------------------+--------------------------------------------------------+---------+-----------------------+---------------+----------+
| Name                   | Description                                            | Type    | Available value       | Default value | Required |
+========================+========================================================+=========+=======================+===============+==========+
| method                 | Method for resampling                                  | string  | "bicubic"             | "bicubic"     | No       |
+------------------------+--------------------------------------------------------+---------+-----------------------+---------------+----------+
| strip_height           | Height of strip (only when tiling is done by strip)    | int     | should be > 0         | 60            | No       |
+------------------------+--------------------------------------------------------+---------+-----------------------+---------------+----------+
| step                   | Horizontal step for resampling inside a strip          | int     | should be > 0         | 500           | No       |
+------------------------+--------------------------------------------------------+---------+-----------------------+---------------+----------+
| interpolator_image     | Interpolation method for the image                     | string  | "bicubic", "nearest"  | "bicubic"     | No       |
+------------------------+--------------------------------------------------------+---------+-----------------------+---------------+----------+
| interpolator_classif   | Interpolation method for the classification            | string  | "bicubic", "nearest"  | "nearest"     | No       |
+------------------------+--------------------------------------------------------+---------+-----------------------+---------------+----------+
| interpolator_mask      | Interpolation method for the mask                      | string  | "bicubic", "nearest"  | "nearest"     | No       |
+------------------------+--------------------------------------------------------+---------+-----------------------+---------------+----------+
| interpolators_edges    | Dict containing interpolation methods for edges        | dict    |                       |               | No       |
+------------------------+--------------------------------------------------------+---------+-----------------------+---------------+----------+
| save_intermediate_data | Save epipolar images and texture                       | boolean |                       | false         | No       |
+------------------------+--------------------------------------------------------+---------+-----------------------+---------------+----------+

Interpolators edges
-------------------

This dict contains an interpolation method for each of the available edge datasets.

+---------------+------------------------------------------+---------+-----------------------+---------------+----------+
| Name          | Description                              | Type    | Available value       | Default value | Required |
+===============+==========================================+=========+=======================+===============+==========+
| edges_mask    | Interpolation method for the edge map    | string  | "bicubic", "nearest"  | "nearest"     | No       |
+---------------+------------------------------------------+---------+-----------------------+---------------+----------+
| depth_map     | Interpolation method for the depth map   | string  | "bicubic", "nearest"  | "bicubic"     | No       |
+---------------+------------------------------------------+---------+-----------------------+---------------+----------+
| normals       | Interpolation method for the normals     | string  | "bicubic", "nearest"  | "bicubic"     | No       |
+---------------+------------------------------------------+---------+-----------------------+---------------+----------+
| tile_id       | Interpolation method for the tile id     | string  | "bicubic", "nearest"  | "nearest"     | No       |
+---------------+------------------------------------------+---------+-----------------------+---------------+----------+

**Example**

.. include-cars-config:: ../../example_configs/configuration/applications_resampling