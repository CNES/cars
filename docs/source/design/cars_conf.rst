================================
CARS configuration: conf module
================================


Detailed requirements
---------------------

* Simple user configuration
* One configuration mechanism
* Be able to be modular (add subparts)
* Definition in each module but be able to load all conf at beginning
* The configuration can describe the pipelines sets to execute.


Solutions
----------

Framework to use:
- Hydra? Seems not the good choice.
- Click (see rasterio)
- Omega only `Slides omega <https://docs.google.com/presentation/d/e/2PACX-1vT_UIV7hCnquIbLUm4NnkUpXvPEh33IKiUEvPRF850WKA8opOlZOszjKdZ3tPmf8u7hGNP6HpqS-NT5/pub?start=false&loop=false&delayms=3000#slide=id.p>`_

Definition of a single all in one configuration (yaml omega example):

.. code-block:: console

    /Global parameters/
    pair1:
      img_left=image_left.tif
      img_right=image_right.tif
    pair2:
      ...

    prepare:

      generate_epipolar_grids: #step1: a function? a sub
          param1:
      sparse_matching:
          param1:
          param2:
      correct_grids:
      ...

    compute_dsm:
      rectification:  #step1 >>> function?
        rectification_internal_param1:  # set to read-only
        rectification_external_param2:  # can be exposed to user

      dense_matching: #step2

      triangulation:

      plugin_name: # add plugin name step. example filter cloud point

      rasterization:




Example of definition in a pipeline:

.. code-block:: console

    prepare:

      generate_epipolar_grids: #step1: a function? a sub
          param1:
      sparse_matching:
          param1:
          param2:
      correct_grids:
      ...

Example of definition in a step:

.. code-block:: console

    rectification:  #step1 >>> function?
      rectification_internal_param1:  # set to read-only
      rectification_external_param2:  # can be exposed to user
