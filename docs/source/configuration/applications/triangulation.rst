.. _triangulation_app:

Triangulation
=============

**Name**: "triangulation"

**Description**

Triangulating the sights and get for each point of the reference image a latitude, longitude, altitude point

**Configuration**

+------------------------+--------------------------------------------------------------------------------------------------------------------+---------+--------------------------------------+------------------------------+----------+
| Name                   | Description                                                                                                        | Type    | Available values                      | Default value               | Required |
+========================+====================================================================================================================+=========+======================================+==============================+==========+
| method                 | Method for triangulation                                                                                           | string  | "line_of_sight_intersection"         | "line_of_sight_intersection" | No       |
+------------------------+--------------------------------------------------------------------------------------------------------------------+---------+--------------------------------------+------------------------------+----------+
| snap_to_img1           | If all pairs share the same left image, modify lines of sight of secondary images to cross those of the ref image  | boolean |                                      | false                        | No       |
+------------------------+--------------------------------------------------------------------------------------------------------------------+---------+--------------------------------------+------------------------------+----------+
| save_intermediate_data | Save depth map as TIF, LAZ and CSV                                                                                 | boolean |                                      | false                        | No       |
+------------------------+--------------------------------------------------------------------------------------------------------------------+---------+--------------------------------------+------------------------------+----------+

**Example**

.. include-cars-config:: ../../example_configs/configuration/applications_triangulation