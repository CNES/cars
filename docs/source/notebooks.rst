=========
Notebooks
=========

Some notebooks are available in the `notebooks` directory of the cars project. They can be used to compute intermediary results and statistics using the cars API.

Notebooks preparation
=====================

Cars has to be installed and a jupyter notebook configuration has be set up.

Quick local installation
------------------------

Use the following automated make command line:

.. code-block:: bash

    make notebook

Advanced installation
---------------------

The automated make command line installs a Jupyter kernel in the virtualenv with the following command:

.. code-block:: bash

    python -m ipykernel install  --sys-prefix --name=cars-venv

and launch a local jupyter notebook environment:

.. code-block:: bash

    jupyter notebook


Notebooks descriptions
======================

`Beware` : Following Notebooks needs preparation step's outputs that have to be generated first and inserted in the notebooks parameters at the beginning.

Compute DSM memory monitoring
-----------------------------

The ``compute_dsm_memory_monitoring.ipynb`` notebook shows how to load data and plot graph to monitor memory consumption during execution of CARS ``compute_dsm`` step with Dask.

The following parameters have to be set :
    * ``compute_dsm_output_dir`` : The output folder of the compute :term:`DSM` step
    * ``nb_workers_per_pbs_jobs`` (Optional) : The number of workers process per pbs job (default : 2)
    * ``nb_pbs_jobs`` : The number of pbs jobs (Number of workers divided by 'nb_workers_per_pbs_jobs')

Epipolar distributions
----------------------

The ``epipolar_distributions.ipynb`` notebook enables to visualize the distributions of the epipolar error and disparity estimated from the matches computed in the preparation step.

The following parameters have to be set :
    * ``cars_home`` : Path to the cars folder.
    * ``content_dir`` :  Path to the directory containing the content.json file of the prepare step output.

low resolution DSM fitting
--------------------------

The ``lowres_dem_fit.ipynb`` notebook details how to estimate and apply the transform to fit A :term:`DSM` to the low resolution initial DEM. This method is currently implemented in cars.

The following parameters have to be set :
    * ``cars_home`` : Path to the cars folder.
    * ``content_dir`` : Path to the directory containing the content.json file of the prepare step output.


Step by step compute DSM
------------------------

The ``step_by_step_compute_dsm.ipynb`` notebook explains how to run step by step :term:`DSM` computation with CARS, starting from the prepare step ouptut folder.

The following parameters have to be set :
    * ``cars_home`` : Path to the cars folder.
    * ``content_dir`` : Path to the directory containing the content.json file of the prepare step output.
    * ``roi_file`` : ROI to process. path to a vector file or raster file as expected by cars. Put roi_file=None to use roi_bbox (mutually exclusive and roi_file has precedence ).
    * ``roi_bbox`` : ROI bounding box of 4 floats to process if roi_file is not defined (None). roi_bbox = [\"xmin\", \"ymin\", \"xmax\", \"ymax\"].
    * ``output_dir`` : Path to output dir where to save figures and data.
