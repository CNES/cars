=========
Notebooks
=========

Some notebooks are available in the ``notebooks`` directory of the CARS project.
CARS pipeline notebook can be used to launch CARS prepare and compute_dsm in a notebook
Following specific notebooks can be used to compute intermediary results and statistics using the CARS API.

Notebooks preparation
=====================

CARS has to be installed and a jupyter notebook configuration has be set up.

Quick local install
-------------------

After installing :ref:`dependencies`, use the following automated make command line:

.. code-block:: console

    make notebook

Advanced install
----------------


The automated ``make notebook`` command line installs CARS with notebook dependencies:

.. code-block:: console

    $ make install-deps        # Install venv virtualenv and CARS dependencies
    $ source venv/bin/activate # Go in the virtualenv
    $ pip install .[notebook]  # Install CARS with notebook dependencies
    or
    $ pip install .             # Install CARS standalone
    $ pip install jupyter bokeh # Install notebook dependencies

Then, a Jupyter kernel can be created in the virtualenv with the following command:

.. code-block:: console

    $ python -m ipykernel install  --sys-prefix --name=cars-venv

Finally, launch a local jupyter notebook environment with:

.. code-block:: console

    $ jupyter notebook


CARS pipelines notebook
=======================

The ``cars_pipelines.ipynb`` notebook show a complete CARS 3D run.
From CARS demo data, it executes the prepare and compute_dsm pipelines from pipeline API.

.. warning::

  The whole CARS kernel described before is needed to launch this notebook.


It depends and includes ``cars_generic.ipynb`` which contains generic functions, demo data configuration.
This generic playbook have to be on the same directory than ``cars_pipelines.ipynb`` notebook.


Specific Notebooks
==================

.. warning::

	 The following notebooks require CARS generated pipeline's outputs (prepare or compute_dsm). These outputs directories have to be inserted in the notebooks beginning configuration.

Step by step compute DSM
------------------------

The ``step_by_step_compute_dsm.ipynb`` notebook explains how to run step by step :term:`DSM` computation with CARS, starting from the prepare step ouptut folder.

The following parameters have to be set :
    * ``cars_home`` : Path to the CARS folder.
    * ``content_dir`` : Path to the directory containing the content.json file of the prepare step output.
    * ``roi_file`` : ROI to process from a file size. It can be either a path to a vector file or a raster file as expected by CARS. Put roi_file=None to use roi_bbox (mutually exclusive and roi_file has precedence ).
    * ``roi_bbox`` : ROI bounding box of 4 floats to process if roi_file is not defined (None). roi_bbox = [\"xmin\", \"ymin\", \"xmax\", \"ymax\"]. They are considered in in final projection depending on EPSG code.
    * ``output_dir`` : Path to output dir where to save figures and data.


Epipolar distributions
----------------------

The ``epipolar_distributions.ipynb`` notebook enables to visualize the distributions of the epipolar error and disparity estimated from the matches computed in the preparation step.

The following parameters have to be set :
    * ``cars_home`` : Path to the CARS folder.
    * ``content_dir`` :  Path to the directory containing the content.json file of the prepare step output.

Low resolution DSM fitting
--------------------------

The ``lowres_dem_fit.ipynb`` notebook details how to estimate and apply the transform to fit a :term:`DSM` to the low resolution initial DEM.
This method is currently implemented in CARS.

The following parameters have to be set :
    * ``cars_home`` : Path to the CARS folder.
    * ``content_dir`` : Path to the directory containing the content.json file of the prepare step output.


Compute DSM memory monitoring
-----------------------------

The ``compute_dsm_memory_monitoring.ipynb`` notebook shows how to load data and plot graph to monitor memory consumption during execution of CARS ``compute_dsm`` step with Dask.

The following parameters have to be set :
    * ``compute_dsm_output_dir`` : The output folder of the compute :term:`DSM` step
    * ``nb_workers_per_pbs_jobs`` (Optional) : The number of workers process per pbs job (default : 2)
    * ``nb_pbs_jobs`` : The number of pbs jobs (Number of workers divided by ``nb_workers_per_pbs_jobs``)
