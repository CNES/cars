.. _grid_generation_app:

Grid Generation
===============

**Name**: "grid_generation"

**Description**

From sensors image, compute the stereo-rectification grids

**Configuration**

+-------------------------+-----------------------------------------------+---------+-----------------------------------+---------------+----------+
| Name                    | Description                                   | Type    |     Available values              | Default value | Required |
+=========================+===============================================+=========+===================================+===============+==========+
| method                  | Method for grid generation                    | string  | "epipolar"                        | epipolar      | No       |
+-------------------------+-----------------------------------------------+---------+-----------------------------------+---------------+----------+
| epi_step                | Step of the deformation grid in nb. of pixels | int     | should be > 0                     | 30            | No       |
+-------------------------+-----------------------------------------------+---------+-----------------------------------+---------------+----------+
| save_intermediate_data  | Save the generated grids                      | boolean |                                   | false         | No       |
+-------------------------+-----------------------------------------------+---------+-----------------------------------+---------------+----------+

**Example**

.. include-cars-config:: ../../example_configs/configuration/applications_grid_generation
