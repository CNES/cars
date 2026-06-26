.. _depth_map_generation_app:

Depth Map Generation
====================

**Name**: "depth_map_generation"

**Description**

Generates a depth map from a single image using the MoGe2 model.

.. warning::

  This application is only available if the `CARS Edge Detection plugin <https://github.com/CNES/cars-edge-detection-plugin>` is installed.

**Configuration**

+------------------------+--------------------------------------------------------+---------+-----------------------+-------------------------------------+----------+
| Name                   | Description                                            | Type    | Available value       | Default value                       | Required |
+========================+========================================================+=========+=======================+=====================================+==========+
| method                 | Method for depth map generation                        | string  | "moge2"               | "moge2"                             | No       |
+------------------------+--------------------------------------------------------+---------+-----------------------+-------------------------------------+----------+
| model                  | Name of the model to use for depth map generation      | string  |                       | "Ruicheng/moge-2-vitl-normal"       | No       |
+------------------------+--------------------------------------------------------+---------+-----------------------+-------------------------------------+----------+
| edge_threshold         | Threshold for edge detection                           | float   | should be >= 0, <= 1  | 0.6                                 | No       |
+------------------------+--------------------------------------------------------+---------+-----------------------+-------------------------------------+----------+
| save_intermediate_data | Save epipolar images and texture                       | boolean |                       | false                               | No       |
+------------------------+--------------------------------------------------------+---------+-----------------------+-------------------------------------+----------+

Model
-----

The model parameter may be any MoGe2 model available on the HuggingFace hub, or any path to a local model.

Officially supported models are:

- Ruicheng/moge-2-vitl-normal
- Ruicheng/moge-2-vitb-normal
- Ruicheng/moge-2-vits-normal

If a model is not already available locally, it will be downloaded from the HuggingFace hub and cached for future use.

The executable ``cars-download-moge2`` can be used to download a model, as further explained in the :ref:`Edge detection example <edge_detection_example>` section.

**Example**

.. include-cars-config:: ../../example_configs/configuration/applications_depth_map_generation