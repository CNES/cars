.. _point_cloud_outlier_removal_app:

Point Cloud outlier removal
===========================

**Name**: "point_cloud_outlier_removal"

**Description**

Point cloud outlier removal

**Configuration**

+------------------------------+------------------------------------------+---------+-----------------------------------+---------------+----------+
| Name                         | Description                              | Type    | Available value                   | Default value | Required |
+==============================+==========================================+=========+===================================+===============+==========+
| method                       | Method for point cloud outlier removal   | string  | "statistical", "small_components" | "statistical" | No       |
+------------------------------+------------------------------------------+---------+-----------------------------------+---------------+----------+
| save_intermediate_data       | Save points clouds as laz and csv format | boolean |                                   | false         | No       |
+------------------------------+------------------------------------------+---------+-----------------------------------+---------------+----------+

If method is *statistical*:

+--------------------+-------------+---------+-----------------+---------------+----------+
| Name               | Description | Type    | Available value | Default value | Required |
+====================+=============+=========+=================+===============+==========+
| k                  |             | int     | should be > 0   | 50            | No       |
+--------------------+-------------+---------+-----------------+---------------+----------+
| filtering_constant |             | float   | should be >= 0  | 0             | No       |
+--------------------+-------------+---------+-----------------+---------------+----------+
| mean_factor        |             | float   | should be >= 0  | 1.3           | No       |
+--------------------+-------------+---------+-----------------+---------------+----------+
| std_dev_factor     |             | float   | should be >= 0  | 3.0           | No       |
+--------------------+-------------+---------+-----------------+---------------+----------+
| use_median         |             | bool    |                 | True          | No       |
+--------------------+-------------+---------+-----------------+---------------+----------+
| half_epipolar_size |             | int     |                 | 5             | No       |
+--------------------+-------------+---------+-----------------+---------------+----------+

If method is *small_components*

+---------------------------------+-------------+---------+-----------------+-----------------+----------+
| Name                            | Description | Type    | Available value | Default value   | Required |
+=================================+=============+=========+=================+=================+==========+
| on_ground_margin                |             | int     |                 | 11              | No       |
+---------------------------------+-------------+---------+-----------------+-----------------+----------+
| connection_distance [#scaled]_  |             | float   |                 | None [#scaled]_ | No       |
+---------------------------------+-------------+---------+-----------------+-----------------+----------+
| nb_points_threshold             |             | int     |                 | 50              | No       |
+---------------------------------+-------------+---------+-----------------+-----------------+----------+
| clusters_distance_threshold     |             | float   |                 | None            | No       |
+---------------------------------+-------------+---------+-----------------+-----------------+----------+
| half_epipolar_size              |             | int     |                 | 5               | No       |
+---------------------------------+-------------+---------+-----------------+-----------------+----------+

.. warning::

    There is a particular case with the *Point Cloud outlier removal* application because both methods can be used at the same time in the pipeline.
    The ninth step consists of Filter the 3D points cloud via N consecutive filters.
    So you can configure the application any number of times. By default the filtering is done twice : once with the *small_components*, once with the *statistical* filter.
    To use your own filters in the order you want, you can add an identifier at the end of each application key :

    * *point_cloud_outlier_removal.my_first_filter*
    * *point_cloud_outlier_removal.filter_2*
    * *point_cloud_outlier_removal.3*

    The filtering steps will then be executed in the order you provided them.

    Because by default the applications *point_cloud_outlier_removal.1* and *point_cloud_outlier_removal.2* are defined, to not do any filtering you must set the configuration of *point_cloud_outlier_removal* to None.

**Examples**

.. include-cars-config:: ../../example_configs/configuration/applications_point_cloud_outlier_removal_1

.. include-cars-config:: ../../example_configs/configuration/applications_point_cloud_outlier_removal_2

.. include-cars-config:: ../../example_configs/configuration/applications_point_cloud_outlier_removal_3

.. rubric:: Footnotes

.. [#scaled] This parameter is computed at runtime depending on the resolution of the input sensor images. You can still override it in the configuration.


