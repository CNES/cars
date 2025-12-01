.. _dense_match_filling_app:

Dense match filling
===================

**Name**: "dense_match_filling"

**Description**

Fill holes in dense matches map.
The zero_padding method fills the disparity with zeros where the selected classification values are non-zero values.

**Configuration**

+-------------------------------------+---------------------------------+-----------+-------------------------+--------------------+----------+
| Name                                | Description                     | Type      | Available value         | Default value      | Required |
+=====================================+=================================+===========+=========================+====================+==========+
| method                              | Method for hole detection       | string    | "zero_padding"          | "zero_padding"     | No       |
+-------------------------------------+---------------------------------+-----------+-------------------------+--------------------+----------+
| save_intermediate_data              | Save disparity map              | boolean   |                         | False              | No       |
+-------------------------------------+---------------------------------+-----------+-------------------------+--------------------+----------+
| classification                      | Values of classes to fill       | List[str] |                         | None               | No       |
+-------------------------------------+---------------------------------+-----------+-------------------------+--------------------+----------+

.. note::
    - The classification of second input is not given. Only the first disparity will be filled with zero value.
    - The filled area will be considered as a valid disparity mask.

**Example**

.. include-cars-config:: ../../example_configs/configuration/applications_dense_match_filling
