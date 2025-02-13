.. _faq:

==========================
Frequently Asked Questions
==========================


Installation
============



CRSError: Not a valid EPSG codes: None
--------------------------------------

As explained in the rasterio FAQ (https://rasterio.readthedocs.io/en/stable/faq.html), rasterio wheels on PyPI include PROJ 7.x and GDAL 3.x.. This can lead to incompatibilities between libraries:

.. code-block:: console

   WARNING :: CPLE_AppDefined in PROJ: internal_proj_create_from_database
   ERROR 1: PROJ: internal_proj_identify [...] proj.db lacks DATABASE.LAYOUT.VERSION.MAJOR / DATABASE.LAYOUT.VERSION.MINOR metadata. It comes from another PROJ installation.
   rasterio.errors.CRSError: Not a valid EPSG codes: None


If GDAL and its dependencies are installed on your computer, we strongly recommend to build rasterio using `--no-binary` option:

.. code-block:: console

   pip install --no-binary rasterio rasterio


Input data
==========

How to create CARS compatible :term:`ROI` input data with OTB ?
---------------------------------------------------------------

Please, see the section :ref:`make_input_roi_images` to generate the ROI input images.


Did you find this error :"The image and the color do not have the same sizes"?
------------------------------------------------------------------------------

If you use Pléiades sensor images, the color image can't be superimposable on the CARS input image.

Please, see the section :ref:`make_a_simple_pan_sharpening` to make a simple pan sharpening with OTB.


How to generate input images in epipolar geometry from grids ?
---------------------------------------------------------------

To generate the images in epipolar geometry from the grids computed by cars and the original images, one can refer to the Orfeo Toolbox documentation `here <https://www.orfeo-toolbox.org/CookBook/recipes/stereo.html#resample-images-in-epipolar-geometry>`_ .


How to make a water mask with gdal on RGBN images?
---------------------------------------------------

Please, see the section :ref:`make_a_water_mask` to make a water mask with OTB.


Output data
===========

How to generate output files overview ?
---------------------------------------

Considering bulky files, it can be recommended to generate an overview file with `GDAL`_ before opening it with `QGIS <https://www.qgis.org>`_:

.. code-block:: console

    gdaladdo -ro -r average dsm.tif 2 4 8 16 32 64
    
Development
===========

Is there a Github ? 
-------------------

Please, see the link `CARS GitHub <https://github.com/CNES/cars>`_ to access CARS github. Do not hesitate to create issues if you have a question or if you encounter a problem.


.. _`GDAL`: https://gdal.org/


