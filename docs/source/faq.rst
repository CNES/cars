.. _faq:

==========================
Frequently Asked Questions
==========================


Installation
============

CARS is correctly installed but I get the following error at runtime:

.. code-block:: console

    ImportError: libvl.so: cannot open shared object file: No such file or directory


**libvl.so** is a dynamic library from VLFeat (see the section :ref:`install`) and it must be placed somewhere in the filesystem.

The GNU standards recommend installing (or copying) by default all libraries in /usr/local/lib (see https://tldp.org/HOWTO/Program-Library-HOWTO/shared-libraries.html).

If your don't have sudo rights, you should add the parent directory of this library to `LD_LIBRARY_PATH` and export it.


Input data
==========

How to create CARS compatible :term:`ROI` input data with OTB ?
---------------------------------------------------------------

Please, see the section :ref:`make_input_roi_images` to generate the ROI input images.


Did you find this error :"The image and the color haven't the same sizes"?
--------------------------------------------------------------------------

If do you use Pleaides sensor images, the color image can't be superimposable on the CARS input image.

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

Considering bulky files, it can be recommended to generate an overview file with `GDAL`_ before opening it with QGIS:

.. code-block:: console

    $ gdaladdo -ro -r average dsm.tif 2 4 8 16 32 64


.. _`GDAL`: https://gdal.org/


