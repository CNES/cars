Mask and Classification Usage
=============================

| Photogrammetry is a technique that cannot reproduce altitude on water. This technique also has difficulties for shaded areas, cloudy areas or moving elements such as cars.
| For this reason, it is possible to mask out areas or apply ad hoc processing to aid the matching stage.


Mask
-----

| CARS can use a mask for each image in order to ignore some image regions. This mask is taken into account during the whole 3D restitution process.

Prerequisites
-------------

- The masks are in the same geometry than the satellite images.
- Each mask is a binary file (one band with 1 nbit). Please, see the section :ref:`convert_image_to_binary_image` to make a binary image.
- 1 values are considered as invalid data. 0 values are considered as valid data.

How it works
-------------

Masks are resampled in epipolar geometry with the resampling application. The masked values (1 values)
are not processed during the computation. They are not taken into account in the matching process
(sparse or dense matching method) to avoid mismatch and useless processing. Furthermore, the sparse
matching estimation of the disparity range can be enhanced with mask using for the water area
typically.

Example 
-------

Below, an example of an area close to Nice with its associated water mask (calculated with SLURP) and the DSM calculated by masking the water zones.

.. list-table:: Example mask usage
   :widths: auto
   :align: center

   * - **Satellite Image**
     - **Water Mask**
     - **Masked DSM**
   * - .. image:: ../images/mask_satellite_nice.png
        :width: 100%
     - .. image:: ../images/mask_watermask_nice.png
        :width: 100%
     - .. image:: ../images/mask_dsm_nice.png
        :width: 100%

Classification
--------------

CARS can use a classification image for each image in order to apply some processings in the regions.
(Please see the section Applications in the Advanced Configuration).

Prerequisites
-------------

- The classification images are in the same geometry as the satellite images.
- Each classification image is a multi-band binary file.

    - Binary file: Please, see the section :ref:`convert_image_to_binary_image` to make a binary image.
    - Multi-band file: One band for each label.
    - 1 values are considered as valid data. 0 values are considered as invalid data.
	
How it works 
------------

By using the classification file, a different application for each band can be applied for the 1
values. See application classification parameter :ref:`advanced configuration`.

Example 
-------

Below, an example of an area close to Nice with its associated classification image (calculated with SLURP). The first band corresponds to the water class and the second band to the vegetation class.

.. list-table:: Example classification usage
   :widths: auto
   :align: center

   * - **Satellite Image**
     - **Classification (Band 1: "Water")**
     - **Classification (Band 2: "Vegetation")**
   * - .. image:: ../images/classif_satellite_nice.png
        :width: 100%
     - .. image:: ../images/classif_watermask_nice.png
        :width: 100%
     - .. image:: ../images/classif_vegetationmask_nice.png
        :width: 100%

