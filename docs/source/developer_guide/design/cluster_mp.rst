.. _cluster_mp:

Cluster Multiprocessing
=======================

Goals
-----

The multiprocessing (MP) cluster facilitates the distribution of computing for the :ref:`application` and the management of :ref:`cars_dataset` data.


Details
-------
The MP cluster is built upon `Python's multiprocessing`_ module using the forkserver mode. In this mode, a pool of worker processes handles the parallel execution of functions. Each worker process is single-threaded, and only essential resources are inherited.
By design, CARS utilizes disk-based registry for data storage, distributing data across the processes.


.. _`Python's multiprocessing`: https://docs.python.org/3/library/multiprocessing.html

How it works
------------

The main class is the MP Cluster, which inherits from the AbstractCluster class. It is instantiated within the orchestrator.

Inspired by the Dask cluster approach, the MP cluster initiates a list of delayed tasks. For each task that has available data (intermediate results input from the linked previous task), the MP cluster transforms the delayed task into an MpFutureTask.

Upon completion of these jobs, the results are saved on disk, and the reference is passed to the next job. The :ref:`refresh_task_cache` function serves as the primary control function of the MP cluster.

The next sections illustrates the architecture of the MP cluster, while the API provides detailed functions that offer more insight into interactions and operations.

Class diagram
^^^^^^^^^^^^^
.. image:: ../../images/mp_cluster.svg
    :align: center

API detailed functions
^^^^^^^^^^^^^^^^^^^^^^

**init**
++++++++
Cluster allocation using a Python thread pool.
The worker pool is set up in forkserver mode with a specified number of workers, job timeouts, and wrapper configuration for cluster logging.

**create_task_wrapped**
+++++++++++++++++++++++
Declare task as **MpDelayed** within the cluster.
**MpDelayed** are instantiated using the **mp_delayed_builder** wrapper builder.
Furthermore, the wrapper provides parameters for the job logger.


**start_tasks**
+++++++++++++++
Add future tasks in the cluster queue. The cluster processes tasks from the queue.
Transform **MpDelayed** with rec_start to **MpJob**, and calculate task dependencies for each job.


**rec_start**
+++++++++++++
Transform delayed tasks to MpJob and create MpFuture objects to retrieve results.

For each task:

1. The function transforms args and kwargs into actual data.

2. Determine the result dependency of the job and verify the status of the ready task.

3. Append the task to the queue with the ready task status, and updated args and kwargs.

4. Create MpFuture to share result object, and remove future at the task's completion.

5. Create additional futures linked to this task.

.. _refresh_task_cache:


**refresh_task_cache**
++++++++++++++++++++++
At each refresh:

1. Sleep (refresh time).

2. Populate the cluster pool with **nb_workers** tasks based on tasks without dependencies. The remaining tasks are added to the **wait_list**.

3. Check for ready results in **in_progress_list**.
   Add job with ready results to **done_list** and map results with statuses in the **done_task_results.**.

   Update/remove dependency for each successfully completed job.

4. Search for next task with higher priority.
   If jobs that have succeeded depend on tasks in the **wait_list**, move these jobs to the **next_priority_tasks** list (ensuring duplicates are removed).

5. Remove completed jobs from the **in_progress_list**.

6. Obtain the lists of **ready_list** and **failed_list** jobs.

7. Filter tasks from the **next_priority_tasks** based on their presence in the **ready_list**, and place them into the **priority_list**.

8. Calculate **nb_ready_task**=**nb_workers** - size(**priority_list**) to add only **nb_ready_task** tasks without dependency.

9. If the priority tasks have completed, proceed with the remaining tasks of the **ready_list** in their initial order.

10. Remove failed jobs from the **wait_list** and copy results to corresponding future, and them remove themselves from **task_cache**.

11. Launch tasks **nb_workers** tasks from **priority_list**.

    Replace jobs with actual data.
    Launch task.
    Eliminate launched tasks from the **wait_list**.

12. Clean unused future jobs with wrapper done_task_results.


**get_ready_failed_tasks**
++++++++++++++++++++++++++
Retrieve the new ready tasks and failed tasks.


**get_tasks_without_deps**
++++++++++++++++++++++++++
A static method evaluates a list of tasks that are ready and lack dependencies, excluding those deemed as initial tasks. 
The initial tasks of the graph have no priority. In order to enhance disk usage efficiency, the cluster initiates with N initial tasks (where N equals the number of workers), assigning priority to the subsequent connected tasks. After finishing a segment of the task graph, the cluster introduces N new initial tasks to continue the process.


**future_iterator**
+++++++++++++++++++
Enable the initiation of all tasks from the orchestrator controller.


**get_job_ids_from_futures**
++++++++++++++++++++++++++++
Obtain a list of job IDs from the future list.

**replace_job_by_data**
+++++++++++++++++++++++
Substitute MpJob instances in lists or dict with their actual data.


**compute_dependencies**
++++++++++++++++++++++++
Compute job result dependencies from args and kw_args.


**MpFutureTask**
++++++++++++++++
A multiprocessing version of the Dask distributed.future.
This class encapsulates data and references to job cluster threads.
It also facilitates the sharing of references between jobs and cleaning cache operations.

**log_error_hook**
++++++++++++++++++
A custom Exception hook to manage cluster thread exceptions.