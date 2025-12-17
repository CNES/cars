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
| nb_match                | number of matches                             | int > 0  |                                   | 100           | No       |
+-------------------------+-----------------------------------------------+----------+-----------------------------------+---------------+----------+

**Example**

**WIP**

.. .. include-cars-config:: ../../example_configs/configuration/applications_grid_generation
