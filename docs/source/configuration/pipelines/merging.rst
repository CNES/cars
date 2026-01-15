.. _merging:

Merging
=======

This pipeline merges several DSMs into a single DSM, including color and
classification layers if provided.


Allowed inputs
--------------

This pipeline takes any number of DSMs as inputs, as explained in the :ref:`DSM input <input_dsm>` section.

Applications
------------

**WIP** : the application(s) in merging have not been created yet.

Advanced Parameters
-------------------

.. list-table::
    :widths: 19 19 19 19
    :header-rows: 1

    * - Name
      - Description
      - Type
      - Default value
    * - save_intermediate_data
      - Save intermediate data for all applications inside this pipeline.
      - bool
      - False
    * - dsm_merging_tile_size
      - Tile size to use in dsms merging
      - int
      - 4000
    * - phasing
      - Phase to use for DSM {"point" : (x,y) , "epsg": epsg}
      - dict
      - None
    * - geometry_plugin
      - Name of the geometry plugin to use and optional parameters (see :ref:`geometry plugin <geometry_plugin>`)
      - str or dict
      - "SharelocGeometry"

Phasing
^^^^^^^
Phase can be added to make sure multiple DSMs can be merged in "dsm -> dsm" pipeline.
"point" and "epsg" of point must be specified

+-------------------+--------------------------+----------------+-------------------------+---------------------------------------+----------+
| Name              | Description              | Type           | Default value           | Available values                      | Required |
+===================+==========================+================+=========================+=======================================+==========+
| *point*           | Point to phase on        | tuple          | None                    |                                       | False    |
+-------------------+--------------------------+----------------+-------------------------+---------------------------------------+----------+
| *epsg*            | Epsg of point            | int            | None                    |                                       | False    |
+-------------------+--------------------------+----------------+-------------------------+---------------------------------------+----------+

.. include-cars-config:: ../../example_configs/pipeline/merging_phasing
