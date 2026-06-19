.. _epipolar_to_sensor_matching_app:

Epipolar to Sensor Matching
===========================

**Name**: "epipolar_to_sensor_matching"

**Description**

Epipolar to Sensor Matching application transforms dispartity maps to sensor matches map, by interpolating the disparity values.
This application is only used if advanced.use_sensor_disp is set to true in surface_modeling configuration.

**Configuration**

+----------------------------+--------------------------------------------------------------------------------------------------------------------+---------+--------------------------------------+------------------------------+----------+
| Name                       | Description                                                                                                        | Type    | Available values                      | Default value               | Required |
+============================+====================================================================================================================+=========+======================================+==============================+==========+
| method                     | Method for epipolar_to_sensor_matching                                                                             | string  | "default"                            | "default"                    | No       |
+----------------------------+--------------------------------------------------------------------------------------------------------------------+---------+--------------------------------------+------------------------------+----------+
| save_intermediate_data     | Save depth map as TIF, LAZ and CSV                                                                                 | boolean |                                      | false                        | No       |
+----------------------------+--------------------------------------------------------------------------------------------------------------------+---------+--------------------------------------+------------------------------+----------+
| tile_size                  | The size of the tiles used for processing                                                                          | integer |                                      | 600                          | No       |
+----------------------------+--------------------------------------------------------------------------------------------------------------------+---------+--------------------------------------+------------------------------+----------+

**Example**

.. include-cars-config:: ../../example_configs/configuration/applications_epipolar_to_sensor_matching