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

The merging pipeline uses these applications:

  - :ref:`dsm_merging <dsm_merging_app>`

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
    * - geometry_plugin
      - Name of the geometry plugin to use and optional parameters (see :ref:`geometry plugin <geometry_plugin>`)
      - str or dict
      - "SharelocGeometry"
