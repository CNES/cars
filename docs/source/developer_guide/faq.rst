==========================
Frequently Asked Questions
==========================


How to use my own geometry library ?
------------------------------------

You can create a new geometry library by inheriting from AbstractGeometry class.
Then, you have to register it in the geometry registry with the decorator.
If you develop it throw a plugin, you have to make sure that the plugin use the right entry point in its setup.py file.
Then, you can select your geometry library in the configuration file.


How to add a new parallelization library ?
------------------------------------------

You can create a new parallelization library by inheriting from AbstractCluster class.
Then, you have to register it in the cluster registry with the decorator.


How to add a new application ?
------------------------------

You can create a new application by inheriting from ApplicationTemplate class.
Then, you have to register it in the application registry with the decorator.
If you develop it throw a plugin, you have to make sure that the plugin use the right entry point in its setup.py file.
Then, you can select your application in the configuration file.

How to add a new method for existing application ?
--------------------------------------------------

You can create a new method by inheriting from the existing application class.
Then, you have to register it in the application registry with the decorator, using the same short_name as the existing application.
If you develop it throw a plugin, you have to make sure that the plugin use the right entry point in its setup.py file.
Then, you can select your method in the configuration file, by using the existing application name.

How to add a new pipeline ?
---------------------------

You can create a new pipeline by inheriting from PipelineTemplate class.
Then, you have to register it in the pipeline registry with the decorator.
If you develop it throw a plugin, you have to make sure that the plugin use the right entry point in its setup.py file.
Then, you can select your pipeline in the configuration file.

Currently, added pipelines can not extend meta pipeline. It should be a standalone pipeline.
You can define you own input data and output data, throw existing or new product levels.

Can I use more than one orchestrator ?
--------------------------------------

A good practice is to use only one orchestrator per pipeline. However, you can use more than one orchestrator if needed.
You need to make sure that only one orchestrator is running tasks at the same time, to avoid overloading machine resources.

Can I use the replacer registry all the time ?
----------------------------------------------

The replacer registry is used to retrieve all task results in memory, in the corresponding CarsDatase. The memory cost can be high depending on the data size.
It is recommended to use it only for small data, or for testing purposes. For instance, you can use it for tie points data, but not for dense matching data.


How to know the corresponding CarsDataset of a computed object, and its position ?
----------------------------------------------------------------------------------

You can get the id of the corresponding CarsDataset by using the method `self.registry.get_future_cars_dataset_id(tile_object)`,
if you added the corresponding CarsDataset to the orchestrator registry (saver_registry or replacer_registry) before computing the task.

In order to get the position of the object in the CarsDataset, you can use the method `self.saver_registry.get_future_cars_dataset_position(tile_object)`.
The position (row, col), must have been added to the future object in the wrapper function, through the saving infos:

.. sourcecode:: python

    # Update the saving info with row and col
    full_saving_info = ocht.update_saving_infos(
        saving_info, row=row, col=col
    )

    ...

    # in the wrapper function, add the saving infos to the data
    cars_dataset.fill_dataset(
        tile_dataset,
        saving_info=full_saving_info,
        window=window,
        profile=None,
        attributes=attributes,
        overlaps=left_overlaps,
    )




Why my orchestrator doesn't compute my tasks ?
----------------------------------------------

If your orchestrator doesn't compute your tasks, it is probably because you didn't add the corresponding CarsDataset to the orchestrator registry (saver_registry or replacer_registry) before computing the task.


Once you have instantiated your CarsDataset, you have to add it to the orchestrator registry, using the methods:
* `self.orchestrator.add_to_save_lists()` for saver_registry
* `self.orchestrator.add_to_replace_lists()` for replacer_registry

Then, as the future tasks are not aware of the CarsDataset they belong to, you have to retrieve cars_dataset id / info:

.. sourcecode:: python

    # Get saving infos in order to save tiles when they are computed
    [
        saving_info
    ] = self.orchestrator.get_saving_infos(
        [your_cars_dataset]
    )


and set it in the data you generate. For instance with Xarray dataset, update attributes:

.. sourcecode:: python

    cars_dataset.fill_dataset(
        xarray_dataset,
        saving_info=saving_info,
    )



How can I reuse tasks already computed ?
----------------------------------------

If CarsDataset with delayed as been added to replacer_registry of orchestrator, once they are computed, the CarsDataset contains the computed data instead of delayed.
However, if it has been added to saver_registry, the CarsDataset still contains delayed, and you have to compute them again.


What is the fastest way to use a new type of data in task, instead of xarray dataset and pandas dataframe ?
-----------------------------------------------------------------------------------------------------------


CarsDict, using json, and store object in it, with post processing function, or add a new type to CarsDataset

If you want to use a new type of data in task, without adding it to the types already supported by CarsDataset (xarray dataset and pandas dataframe), you can use CarsDict, that allows to store any type of object in a json file.
However, the object you store in CarsDict must be serializable in json format.

How to create a CarsDataset for non ordered tiles ?
---------------------------------------------------

If your tiles are not ordered, you can create a CarsDataset with unordered tiles, with the dimensions tou want: (N, M), (NxM, 1), (1, NxM), etc.