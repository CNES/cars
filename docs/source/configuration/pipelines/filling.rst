.. _filling:

Filling
=======

This pipeline fills holes or masked regions in the DSM, using configurable interpolation or
reconstruction methods. 

Allowed inputs
--------------

This pipeline takes a single DSM as input, as explained in the :ref:`DSM input <input_dsm>` section.

.. warning::

  In order for filling to work, the DSM needs to have been generated with the dense match filling ``zero_padding`` method used during the `surface_modeling` pipeline.

Applications
------------

This pipeline uses the following applications : 

- :ref:`dsm_filling <dsm_filling_app>`
- :ref:`auxiliary_filling <auxiliary_filling_app>`

The DSM filling application can be called multiple times, as shown in the application's documentation page.

Advanced Parameters
-------------------

.. list-table::
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
