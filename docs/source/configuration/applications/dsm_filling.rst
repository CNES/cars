.. _dsm_filling_app:

DSM Filling
===========

**Name**: "dsm_filling"

**Description**

Fill classified values or missing values with one the three avalable methods.

**Configuration**

+-------------------------------------+---------------------------------+---------+----------------------------------------------------------+--------------------+----------+
| Name                                | Description                     | Type    | Available value                                          | Default value      | Required |
+=====================================+=================================+=========+==========================================================+====================+==========+
| method                              | Method for hole detection       | string  | "exogenous_filling", "bulldozer", "border_interpolation" |                    | Yes      |
+-------------------------------------+---------------------------------+---------+----------------------------------------------------------+--------------------+----------+
| save_intermediate_data              | Save disparity map              | boolean |                                                          | False              | No       |
+-------------------------------------+---------------------------------+---------+----------------------------------------------------------+--------------------+----------+


**Method exogenous_filling:**

Method "exogenous_filling" fills with altitude of exogenous data (DEM/geoid).

+-------------------------------------+----------------------------------------------------+-------------+-------------------------+--------------------+----------+
| Name                                | Description                                        | Type        | Available value         | Default value      | Required |
+=====================================+====================================================+=============+=========================+====================+==========+
| classification                      | Values of classes to fill                          | List[str]   |                         | "nodata"           | No       |
+-------------------------------------+----------------------------------------------------+-------------+-------------------------+--------------------+----------+
| fill_with_geoid                     | Classes to fill with geoid                         | List[str]   |                         | None               | No       |
+-------------------------------------+----------------------------------------------------+-------------+-------------------------+--------------------+----------+
| interpolation_method                | Interpolation method for DEM and geoid resampling  | List[str]   | "bilinear", "cubic"     | None               | No       |
+-------------------------------------+----------------------------------------------------+-------------+-------------------------+--------------------+----------+


**Method bulldozer:**

Method "bulldozer" converts the DSM to a DTM and fills the pixels with the output DTM.

+-------------------------------------+---------------------------------+-----------+-------------------------+--------------------+----------+
| Name                                | Description                     | Type      | Available value         | Default value      | Required |
+=====================================+=================================+===========+=========================+====================+==========+
| classification                      | Values of classes to fill       | List[str] |                         | "nodata"           | No       |
+-------------------------------------+---------------------------------+-----------+-------------------------+--------------------+----------+

**Method border_interpolation:**

Method "border_interpolation" use the border of every component to compute the altitude to fill.

+-------------------------------------+------------------------------------------+-----------+-------------------------+--------------------+----------+
| Name                                | Description                              | Type      | Available value         | Default value      | Required |
+=====================================+==========================================+===========+=========================+====================+==========+
| classification                      | Values of classes to fill                | List[str] |                         | "nodata"           | No       |
+-------------------------------------+------------------------------------------+-----------+-------------------------+--------------------+----------+
| component_min_size                  | Minimal size (pixels) of feature to fill | int       |                         | 5                  | No       |
+-------------------------------------+------------------------------------------+-----------+-------------------------+--------------------+----------+
| border_size                         | Size of border used to estimate altitude | int       |                         | 10                 | No       |
+-------------------------------------+------------------------------------------+-----------+-------------------------+--------------------+----------+
| percentile                          | Percentile of border taken for altitude  | float     |                         | 10                 | No       |
+-------------------------------------+------------------------------------------+-----------+-------------------------+--------------------+----------+

.. note::
    - If the keyword "nodata" is added to the classification parameter, nodata pixels of the classification will be filled. If no classification is given, nodata pixels of DSM will be filled.

.. warning::

    There is a particular case with the *dsm_filling* application because it can be called any number of times.
    Because it is not possible to define three times the *dsm_filling* in your yaml/json configuration file, you can add an identifier after *dsm_filling* to differentiate each application :

    * *dsm_filling.border_interp*
    * *dsm_filling.two*
    * *dsm_filling.with_bulldozer*

    It is recommended to run bulldozer before border_interpolation in order for border_interpolation to get a DTM. If no DTM is found, border_interpolation will use the DSM.
    The execution order is determined by the order of the applications in the configuration file.

**Example**

.. include-cars-config:: ../../example_configs/configuration/applications_dsm_filling_1

.. include-cars-config:: ../../example_configs/configuration/applications_dsm_filling_2

.. include-cars-config:: ../../example_configs/configuration/applications_dsm_filling_3