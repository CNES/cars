.. _orchestrator:

Orchestrator
============

Goals
-----

The *orchestrator* is the central element of CARS concepts.
Its role is to ensure the communication between the *computing technology*, the *applications* and the *CarsDatasets*.

Details
-------

The *orchestrator* is unique and instantiated for each pipeline:


.. sourcecode:: python

        with orchestrator.Orchestrator(distributed_conf=distributed_conf) as cars_orchestrator:


It is mainly composed of:

* a *cluster*
* a *CarsDatasetRegistry*
* dictionary for output json file containing information given by applications

.. _cluster:

Cluster
^^^^^^^

The cluster is the component which allows to realize the calculations.

.. sourcecode:: python

    class AbstractCluster(metaclass=ABCMeta):


        ...

        @abstractmethod
        def create_task(self, func, nout=1):
            """
            Create task

            :param func: function
            :param nout: number of outputs
            """

        @abstractmethod
        def start_tasks(self, task_list):
            """
            Start all tasks

            :param task_list: task list
            """

        @abstractmethod
        def future_iterator(self, future_list):
            """
            Iterator, iterating on computed futures

            :param future_list: future_list list
            """



 The two main functions are:

* `create_task` to declare a task to the cluster. It returns `delayed` object.
* `start_tasks` to compute each task that have been declared.
* `future_iterator`: iterate over the `future` objects

There are already 3 plugins, each one representing a mode:

* *dask*

   * *local_dask*
   * *pbs_dask*
   * *slurm_dask*

* *mp* (for mutliprocessing)
* *sequential* : (note: `delayed` is note a real one, it is directly the data type, so `Xarray.dataset` or `Panda.Dataframe`)


CarsDatasetRegistry
^^^^^^^^^^^^^^^^^^^^^

The *CarsDatasetRegistry* is a class that allows to manage the list of *CarsDatasets* that user wants to save.
It is mainly composed of:

* a registry *CarsDataset* list
* id associated to each registered *CarsDataset*

There is some functions that allows to:

* Add new *CarsDataset* to registry
* Obtain an ID for a CarsDataset
* Find a *CarsDataset* from an ID
* Manage saving tile by tile (i.e future by future, related to `dask` terms), by using the `SingleCarsDatasetSaver` that wraps `CarsDataset` save functions.


How it works
^^^^^^^^^^^^

1. Instantiate *orchestrator* before every pipeline with configuration file for defining cluster mode and output directory

.. sourcecode:: python

        with orchestrator.Orchestrator(distributed_conf=distributed_conf) as cars_orchestrator:


*Cluster* and *CarsDatasetRegistry* are created

.. sourcecode:: python

    def __init__(self, distributed_conf=None):

        """
        Init function of Orchestrator.
        Creates Cluster and Registry for CarsDatasets

        :param distributed_conf: configuration of distribution
        """

        # out_dir
        self.out_dir = None
        if "out_dir" in distributed_conf:
            self.out_dir = distributed_conf["out_dir"]
        else:
            logging.error("No out_dir defined")

        self.nb_workers = 1
        if "nb_workers" in distributed_conf:
            self.nb_workers = distributed_conf["nb_workers"]

        # init cluster
        self.cluster = AbstractCluster(  # pylint: disable=E0110
            distributed_conf
        )

        # init CarsDataset savers registry
        self.cars_ds_savers_registry = CarsDatasetsRegistry()

        # init saving lists
        self.cars_ds_list = []

        # outjson
        self.out_json = {}


2. *Orchestrator* is used in every applications which can add *CarsDataset* to save (*orchestrator* interacts with *CarsDatasetRegistry*)

.. sourcecode:: python

    def add_to_save_lists(
        self, file_name, tag, cars_ds, dtype="float32", nodata=0
    ):
        """
        Save file to list in order to be saved later

        :param file_name: file name
        :param tag: tag
        :param cars_ds: cars dataset to register
        """

        self.cars_ds_savers_registry.add_file_to_save(
            file_name, cars_ds, tag=tag, dtype=dtype, nodata=nodata
        )

        if cars_ds not in self.cars_ds_list:
            self.cars_ds_list.append(cars_ds)


3. *Orchestrator* can be used to obtain *CarsDataset* ID (see :ref:`application`)

.. sourcecode:: python

    def get_saving_infos(self, cars_ds_list):
        """
        Get saving infos of given cars datasets

        :param cars_ds_list: list of cars datasets
        :type cars_ds_list: list[CarsDataset]

        :return : list of saving infos
        :rtype: list[dict]
        """

        saving_infos = []

        for cars_ds in cars_ds_list:
            saving_infos.append(
                self.cars_ds_savers_registry.get_saving_infos(cars_ds)
            )

        return saving_infos

4. At the end of the pipeline, the `__exit__` function is called automatically. It computes all `delayed` needed for saving *CarsDataset*
using `cluster.start_tasks` function` that returns `future` objects.
And the `save` function of *CarsDatasetRegistry* is called for saving by iterating on `future` objects.

.. sourcecode:: python

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Function run on exit.

        Compute cluster tasks, save futures to be saved, and cleanup cluster
        and files

        """

        # run compute and save files
        logging.info("Compute delayed ...")
        # Flatten to list
        delayed_objects = flatten_object(
            self.cars_ds_savers_registry.get_cars_datasets_list()
        )

        # Compute delayed
        future_objects = self.cluster.start_tasks(delayed_objects)

        # Save objects when they are computed
        logging.info("Wait for futures results ...")
        pbar = tqdm(total=len(future_objects), desc="Processing Futures ...")
        for future_obj in tqdm(self.cluster.future_iterator(future_objects)):
            # get corresponding CarsDataset and save tile
            if future_obj is not None:
                self.cars_ds_savers_registry.save(future_obj)
            else:
                logging.debug("None tile : not saved")
            pbar.update()

        # close files
        logging.info("Close files ...")
        self.cars_ds_savers_registry.cleanup()

        # close cluster
        logging.info("Close cluster ...")
        self.cluster.cleanup()
