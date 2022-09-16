.. _orchestrator_config:

============
Orchestrator
============

The chain have computing distribution capabilities and can use dask (local or distributed cluster) or multiprocessing libraries to distribute the computations.
The distributed cluster require centralized files storage and uses PBS scheduler only for now.

This key is optional and allows to define orchestrator configuration that controls the distributed computations:

+------------------+-----------------------------------------------------------+-----------------------------------------+---------------+----------+
| Name             | Description                                               | Type                                    | Default value | Required |
+==================+===========================================================+=========================================+===============+==========+
| *mode*           | Parallelization mode "local_dask", "pbs_dask" or "mp"     | string                                  |local_dask     | No       |
+------------------+-----------------------------------------------------------+-----------------------------------------+---------------+----------+
| *nb_workers*     | Number of workers                                         | int, should be > 0                      | 2             | No       |
+------------------+-----------------------------------------------------------+-----------------------------------------+---------------+----------+
| *walltime*       | Walltime for one worker                                   | string, Should be formatted as HH:MM:SS | 00:59:00      | No       |
+------------------+-----------------------------------------------------------+-----------------------------------------+---------------+----------+