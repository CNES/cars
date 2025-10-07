.. _orchestrator_basic:

Orchestrator (basic)
====================

The ``orchestrator`` key is optional and allows to define orchestrator configuration that controls the distribution. 

The orchestrator can be higyly configured but this section will only show a basic configuration of orchestrator in multiprocessing mode to scale manually the amount of CPU and RAM used by CARS.

By default, orchestrator mode is set to `auto` : the number of CPU and amount of memory is automatically set according to the machine specifications.

At least 2000 Mb of RAM must be available to run CARS in auto mode.

But you also can use multiprocessing mode and fill the parameters *nb_workers* and *max_ram_per_worker* according to the resources you requested.

+----------------------+--------------------------------------------------------------------------------------+-----------------+-----------------+----------+
| Name                 | Description                                                                          | Type            | Default value   | Required |
+======================+======================================================================================+=================+=================+==========+
| *mode*               | Parallelization mode (set it to "multiprocessing" to edit the following parameters)  | string          | "auto"          | Yes      |
+----------------------+--------------------------------------------------------------------------------------+-----------------+-----------------+----------+
| *nb_workers*         | Number of workers                                                                    | int             | 2               | No       |
+----------------------+--------------------------------------------------------------------------------------+-----------------+-----------------+----------+
| *max_ram_per_worker* | Maximum ram per worker                                                               | int, float      | 2000            | No       |
+----------------------+--------------------------------------------------------------------------------------+-----------------+-----------------+----------+

For example, il you want to limit CARS resources to 6 CPU and 18 Go of RAM, you can configure the orchestrator as follows : 

.. include-cars-config:: ../../example_configs/how_to_use_CARS/basic_configuration/orchestrator_basic_config

More parameters for the ``orchestrator`` section are documented in :ref:`orchestrator_advanced`