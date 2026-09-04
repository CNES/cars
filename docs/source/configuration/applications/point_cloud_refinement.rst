.. _point_cloud_refinement_app:

Point Cloud Refinement
======================

**Name**: "point_cloud_refinement"

**Description**

Refine a triangulated point cloud using target surface normals and displacement optimization along the normals.
This application improves the geometric consistency of the 3D point cloud by aligning it with target normal constraints.
A global rotation is estimated from the surface geometry to align target normals with the XYZ-derived reference frame before refinement.

This application is used when a normal map was given as input, either through the edge detection plugin or manually from the configuration file.
This application is skipped otherwise.

**Configuration**

+------------------------------+----------------------------------------------------------+---------+-----------------------------------+------------------+----------+
| Name                         | Description                                              | Type    | Available value                   | Default value    | Required |
+==============================+==========================================================+=========+===================================+==================+==========+
| method                       | Method for point cloud refinement                        | string  | "normals_guided"                  | "normals_guided" | No       |
+------------------------------+----------------------------------------------------------+---------+-----------------------------------+------------------+----------+
| activated                    | Run this application (if false, skip processing)         | boolean |                                   | true             | No       |
+------------------------------+----------------------------------------------------------+---------+-----------------------------------+------------------+----------+
| save_intermediate_data       | Save the refined point cloud and displacement map as TIF | boolean |                                   | false            | No       |
+------------------------------+----------------------------------------------------------+---------+-----------------------------------+------------------+----------+

If method is *normals_guided*:

+------------------------------+----------------------------------------------------------------------------------------------------------+---------+-----------------------------------+---------------+----------+
| Name                         | Description                                                                                              | Type    | Available value                   | Default value | Required |
+==============================+==========================================================================================================+=========+===================================+===============+==========+
| w_guidance                   | Weight of normal alignment to the target normals                                                         | float   | >= 0                              | 1.0           | No       |
+------------------------------+----------------------------------------------------------------------------------------------------------+---------+-----------------------------------+---------------+----------+
| w_smooth                     | Weight of displacement smoothness (higher -> more gradual displacement from one pixel to the next)       | float   | >= 0                              | 0.0           | No       |
+------------------------------+----------------------------------------------------------------------------------------------------------+---------+-----------------------------------+---------------+----------+
| w_data                       | Weight of triangulation data attachment                                                                  | float   | >= 0                              | 0.2           | No       |
+------------------------------+----------------------------------------------------------------------------------------------------------+---------+-----------------------------------+---------------+----------+
| max_displacement             | Maximum displacement allowed (meters)                                                                    | float   | >= 0                              | 0.5           | No       |
+------------------------------+----------------------------------------------------------------------------------------------------------+---------+-----------------------------------+---------------+----------+
| update_xy                    | Update XY coordinates alongside Z coordinates                                                            | boolean |                                   | true          | No       |
+------------------------------+----------------------------------------------------------------------------------------------------------+---------+-----------------------------------+---------------+----------+
| use_metric                   | Solve in metric frame (EPSG 4978)                                                                        | boolean |                                   | true          | No       |
+------------------------------+----------------------------------------------------------------------------------------------------------+---------+-----------------------------------+---------------+----------+
| damp                         | Damping coefficient for LSQR solver                                                                      | float   | >= 0                              | 1.0e-6        | No       |
+------------------------------+----------------------------------------------------------------------------------------------------------+---------+-----------------------------------+---------------+----------+
| max_iter                     | Maximum iterations for LSQR solver                                                                       | integer | > 0                               | 20            | No       |
+------------------------------+----------------------------------------------------------------------------------------------------------+---------+-----------------------------------+---------------+----------+

**Outputs**

The refined point cloud is saved with the following bands:

- ``X_refined.tif``: Refined X coordinates
- ``Y_refined.tif``: Refined Y coordinates
- ``Z_refined.tif``: Refined Z coordinates
- ``displacement.tif``: 3-band displacement map (dx, dy, dz; in meters)
- ``invalidity_mask_refined.tif``: Mask of points refined

**Example**

.. include-cars-config:: ../../example_configs/configuration/applications_point_cloud_refinement
