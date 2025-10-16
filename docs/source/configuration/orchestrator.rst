.. _orchestrator:

Orchestrator
============

CARS can distribute the computations chunks by using either dask (local or distributed cluster) or multiprocessing libraries.
The distributed cluster require centralized files storage and uses PBS scheduler.

The ``orchestrator`` key is optional and allows to define orchestrator configuration that controls the distribution:

+------------------+----------------------------------------------------------------------------------------------------------+-----------------------------------------+-----------------+----------+
| Name             | Description                                                                                              | Type                                    | Default value   | Required |
+==================+==========================================================================================================+=========================================+=================+==========+
| *mode*           | Parallelization mode "local_dask", "pbs_dask", "slurm_dask", "multiprocessing", "auto" or "sequential"   | string                                  | "auto"          | Yes      |
+------------------+----------------------------------------------------------------------------------------------------------+-----------------------------------------+-----------------+----------+
| *task_timeout*   | Time (seconds) betweend two tasks before closing cluster and restarting tasks                            | int                                     | 600             | No       |
+------------------+----------------------------------------------------------------------------------------------------------+-----------------------------------------+-----------------+----------+
| *profiling*      | Configuration for CARS profiling mode                                                                    | dict                                    |                 | No       |
+------------------+----------------------------------------------------------------------------------------------------------+-----------------------------------------+-----------------+----------+

.. note::
    `sequential` orchestrator purposes are mostly for studies, debug and notebooks. If you want to use it with large data, consider using a ROI and Epipolar A Priori. Only tiles needed for the specified ROI will be computed. If Epipolar A priori is not specified, Epipolar Resampling and Sparse Matching will be performed on the whole image, no matter what ROI field is filled with.

.. note::
    `auto` mode is a shortcut for *multiprocessing* orchestrator with parameters *nb_workers* and *max_ram_per_worker* are set:
    * *max_ram_per_worker* : 2000
    * *nb_workers* : Computed accordingly to the available RAM.

    At least 2000 Mb of RAM must be available to run CARS in auto mode.

    In this case, use multiprocessing mode and fill the parameters *nb_workers* and *max_ram_per_worker* according to the resources you requested.


Depending on the used orchestrator mode, the following parameters can be added in the configuration:

**Mode local_dask, pbs_dask:**

+---------------------+------------------------------------------------------------------+-----------------------------------------+---------------+----------+
| Name                | Description                                                      | Type                                    | Default value | Required |
+=====================+==================================================================+=========================================+===============+==========+
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


**Mode multiprocessing:**

+-----------------------+-----------------------------------------------------------------+------------------------------------------+---------------+----------+
| Name                  | Description                                                     | Type                                     | Default value | Required |
+=======================+=================================================================+==========================================+===============+==========+
| *mp_mode*             | The type of multiprocessing mode "forkserver", "fork", "spawn"  | str                                      | "forkserver"  | No       |
+-----------------------+-----------------------------------------------------------------+------------------------------------------+---------------+----------+
| *nb_workers*          | Number of workers                                               | int, should be > 0                       | 2             | No       |
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

    **Factorisation**

    Two or more tasks are sequentially dependant if they can be run sequentially, independantly from any other task.
    If it is the case, those tasks can be factorized, which means they can be run in a single task.

    Running several tasks in one task avoids doing useless dumps on disk between sequential tasks. It does not lose time
    because tasks that are factorized could not be run in parallel, and it permits to save some time from the
    creation of tasks and data transfer that are avoided.

.. note::

    If you are working on windows, the spawn multiprocessing mode has to be used. If you are putting "fork" or "forkserver", it will be forced to spawn.

**Profiling configuration:**

The profiling mode is used to analyze time or memory of the executed CARS functions at worker level. By default, the profiling mode is disabled.
It could be configured for the different orchestrator modes and for different purposes (time, elapsed time, memory allocation, loop testing).

.. include-cars-config:: ../example_configs/configuration/orchestrator

+---------------------+-----------------------------------------------------------+-----------------------------------------+----------------+----------+
| Name                | Description                                               | Type                                    | Default value  | Required |
+=====================+===========================================================+=========================================+================+==========+
| *mode*              | type of profiling mode "cars_profiling, cprofile, memray" | string                                  | cars_profiling | No       |
+---------------------+-----------------------------------------------------------+-----------------------------------------+----------------+----------+
| *loop_testing*      | enable loop mode to execute each step multiple times      | bool                                    | False          | No       |
+---------------------+-----------------------------------------------------------+-----------------------------------------+----------------+----------+

- Please use make command 'profile-memory-report' to generate a memory profiling report from the memray outputs files (after the memray profiling execution).
- Please disabled profiling to eval memory profiling at master orchestrator level and execute make command instead: 'profile-memory-all'.

.. note::

    The logging system provides messages for all orchestration modes, both for the main process and the worker processes.
    The logging output file of the main process is located in the output directory.
    In the case of distributed orchestration, the worker's logging output file is located in the workers_log directory (the message format indicates thread ID and process ID).
    A summary of basic profiling is generated in output directory.
