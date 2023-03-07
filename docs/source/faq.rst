.. _faq:

==========================
Frequently Asked Questions
==========================


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


