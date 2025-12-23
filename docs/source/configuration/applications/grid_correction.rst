.. _grid_correction_app:

Grid Correction
===============

**Name**: "grid_correction"

**Description**

Using stereo-rectification grids and sparse matches (i.e provided by Tiepoints), correct the grid for any global deformation.

**Configuration**

**WIP**

+-------------------------+-----------------------------------------------+----------+-----------------------------------+---------------+----------+
| Name                    | Description                                   | Type     |     Available values              | Default value | Required |
+=========================+===============================================+==========+===================================+===============+==========+
| method                  | Method for grid correction                    | string   | default                           | default       | No       |
+-------------------------+-----------------------------------------------+----------+-----------------------------------+---------------+----------+
| save_intermediate_data  | Save the generated grids                      | boolean  |                                   | false         | No       |
+-------------------------+-----------------------------------------------+----------+-----------------------------------+---------------+----------+
| nb_matches              | number of matches required                    | int      |  should be > 0                    | 90            | No       |
+-------------------------+-----------------------------------------------+----------+-----------------------------------+---------------+----------+

**Example**

.. .. include-cars-config:: ../../example_configs/configuration/applications_grid_correction
