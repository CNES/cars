
===============
Troubleshooting
===============


Known Issues
============

Problems installing CARS on system with proj and gdal
-----------------------------------------------------

Rasterio and Fiona must be compiled with your version of proj and gdal. Pip install these packages in no-binary mode.


My job is killed using Multiprocessing
--------------------------------------

It can happen for several reasons:

* Out of memory: check the profiling graphs to see if CARS consumes the memory you were expecting.
* SegFault in worker: If a segmentation fault happens in a worker, in a binded c++ code, no error message is displayed.


Dask workers die
----------------

Dask workers can die for several reasons:
* Out of memory in worker
* unmanaged memory can increase and kill the worker




Debugging Tips
==============


A lot of information in contained in the logs `outdir/logs/*log` files. Moreover, a profiling report is generated, and can explain the majority of crashes.

For quality issues, you should visualize intermediate results, that can be saved by activating the `save_intermediate_data` option in the configuration file of each application.


When the crashes occur in a worker, it can be tricky to debug. You can try the following tips:
* Change the cluster type to `squential` to have logs in the main process. Be careful, this can be very slow : use it on small datasets only.
* Use the roi option to reduce the data size.

