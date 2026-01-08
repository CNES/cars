.. _orchestrator:

Orchestrator
============

CARS can distribute the computations chunks by using either dask (local or distributed cluster) or multiprocessing libraries.
The distributed cluster require centralized files storage and uses PBS scheduler.

The ``orchestrator`` key is optional and allows to define orchestrator configuration that controls the distribution:

+------------------+----------------------------------------------------------------------------------------------------------+-----------------------------------------+-------------------+----------+
| Name             | Description                                                                                              | Type                                    | Default value     | Required |
+==================+==========================================================================================================+=========================================+===================+==========+
| *mode*           | Parallelization mode "multiprocessing", "local_dask", "pbs_dask", "slurm_dask" or "sequential"           | string                                  | "multiprocessing" | Yes      |
+------------------+----------------------------------------------------------------------------------------------------------+-----------------------------------------+-------------------+----------+
| *task_timeout*   | Time (seconds) betweend two tasks before closing cluster and restarting tasks                            | int                                     | 600               | No       |
+------------------+----------------------------------------------------------------------------------------------------------+-----------------------------------------+-------------------+----------+

.. note::
    `sequential` orchestrator purposes are mostly for studies, debug and notebooks. If you want to use it with large data, consider using a ROI and Epipolar A Priori. Only tiles needed for the specified ROI will be computed. If Epipolar A priori is not specified, Epipolar Resampling and Sparse Matching will be performed on the whole image, no matter what ROI field is filled with.


Depending on the used orchestrator mode, the following parameters can be added in the configuration:



**Mode multiprocessing:**

+-----------------------+-----------------------------------------------------------------+------------------------------------------+---------------+----------+
| Name                  | Description                                                     | Type                                     | Default value | Required |
+=======================+=================================================================+==========================================+===============+==========+
| *mp_mode*             | The type of multiprocessing mode "forkserver", "fork", "spawn"  | str                                      | "forkserver"  | No       |
+-----------------------+-----------------------------------------------------------------+------------------------------------------+---------------+----------+
| *nb_workers*          | Number of workers : "auto" of int.                              | int, should be > 0                       | "auto"        | No       |
+-----------------------+-----------------------------------------------------------------+------------------------------------------+---------------+----------+
| *max_ram_per_worker*  | Maximum ram per worker                                          | int or float, should be > 0              | 2000          | No       |
+-----------------------+-----------------------------------------------------------------+------------------------------------------+---------------+----------+
| *max_tasks_per_worker*| Number of tasks a worker can complete before refresh            | int, should be > 0                       | 10            | No       |
+-----------------------+-----------------------------------------------------------------+------------------------------------------+---------------+----------+
| *dump_to_disk*        | Dump temporary files to disk                                    | bool                                     | True          | No       |
+-----------------------+-----------------------------------------------------------------+------------------------------------------+---------------+----------+
| *per_job_timeout*     | Timeout used for a job                                          | int or float                             | 600           | No       |
+-----------------------+-----------------------------------------------------------------+------------------------------------------+---------------+----------+
| *factorize_tasks*     | Tasks sequentially dependent are run in one task                | bool                                     | True          | No       |
+-----------------------+-----------------------------------------------------------------+------------------------------------------+---------------+----------+

.. note::

   "auto" for *nb_workers* uses the number of CPU cores available on the machine minus one (to keep one core for the main process), without exceeding 50% of the available RAM (*max_ram_per_worker* x *nb_workers* < total RAM / 2).


.. note::

    **Factorisation**

    Two or more tasks are sequentially dependant if they can be run sequentially, independantly from any other task.
    If it is the case, those tasks can be factorized, which means they can be run in a single task.

    Running several tasks in one task avoids doing useless dumps on disk between sequential tasks. It does not lose time
    because tasks that are factorized could not be run in parallel, and it permits to save some time from the
    creation of tasks and data transfer that are avoided.

.. note::

    If you are working on windows, the spawn multiprocessing mode has to be used. If you are putting "fork" or "forkserver", it will be forced to spawn.


**Mode local_dask, pbs_dask:**

+---------------------+------------------------------------------------------------------+-----------------------------------------+---------------+----------+
| Name                | Description                                                      | Type                                    | Default value | Required |
+=====================+==================================================================+=========================================+===============+==========+
| *nb_workers*        | Number of workers                                                | string or int, should be > 0            | 2             | No       |
+---------------------+------------------------------------------------------------------+-----------------------------------------+---------------+----------+
| *max_ram_per_worker*| Maximum ram per worker                                           | int or float, should be > 0             | 2000          | No       |
+---------------------+------------------------------------------------------------------+-----------------------------------------+---------------+----------+
| *walltime*          | Walltime for one worker                                          | string, Should be formatted as HH:MM:SS | 00:59:00      | No       |
+---------------------+------------------------------------------------------------------+-----------------------------------------+---------------+----------+
| *use_memory_logger* | Usage of dask memory logger                                      | bool, True if use memory logger         | False         | No       |
+---------------------+------------------------------------------------------------------+-----------------------------------------+---------------+----------+
| *activate_dashboard*| Usage of dask dashboard                                          | bool, True if use dashboard             | False         | No       |
+---------------------+------------------------------------------------------------------+-----------------------------------------+---------------+----------+
| *python*            | Python path to binary to use in workers (not used in local dask) | str                                     | Null          | No       |
+---------------------+------------------------------------------------------------------+-----------------------------------------+---------------+----------+


**Mode slurm_dask:**

+---------------------+------------------------------------------------------------------+-----------------------------------------+---------------+----------+
| Name                | Description                                                      | Type                                    | Default value | Required |
+=====================+==================================================================+=========================================+===============+==========+
| *account*           | SLURM account                                                    | str                                     |               | Yes      |
+---------------------+------------------------------------------------------------------+-----------------------------------------+---------------+----------+
| *nb_workers*        | Number of workers                                                | int, should be > 0                      | 2             | No       |
+---------------------+------------------------------------------------------------------+-----------------------------------------+---------------+----------+
| *max_ram_per_worker*| Maximum ram per worker                                           | int or float, should be > 0             | 2000          | No       |
+---------------------+------------------------------------------------------------------+-----------------------------------------+---------------+----------+
| *walltime*          | Walltime for one worker                                          | string, Should be formatted as HH:MM:SS | 00:59:00      | No       |
+---------------------+------------------------------------------------------------------+-----------------------------------------+---------------+----------+
| *use_memory_logger* | Usage of dask memory logger                                      | bool, True if use memory logger         | False         | No       |
+---------------------+------------------------------------------------------------------+-----------------------------------------+---------------+----------+
| *activate_dashboard*| Usage of dask dashboard                                          | bool, True if use dashboard             | False         | No       |
+---------------------+------------------------------------------------------------------+-----------------------------------------+---------------+----------+
| *python*            | Python path to binary to use in workers (not used in local dask) | str                                     | Null          | No       |
+---------------------+------------------------------------------------------------------+-----------------------------------------+---------------+----------+
| *qos*               | Quality of Service parameter (qos list separated by comma)       | str                                     | Null          | No       |
+---------------------+------------------------------------------------------------------+-----------------------------------------+---------------+----------+

