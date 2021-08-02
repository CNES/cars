================================
CARS configuration : conf module
================================


Detailed requirements
---------------------

* Simple user configuration
* One configuration mechanism
* Be able to be modular (add subparts)
* Definition in each module but be able to load all conf at beginning
* The configuration can describe the pipelines sets to execute. 


Solutions
---------

Framework to use:
- Hydra ?
- Click (see rasterio)
- ...

Definition of a single all in one configuration:

.. code-block:: console

    /Global parameters/






Example of definition in each module:
