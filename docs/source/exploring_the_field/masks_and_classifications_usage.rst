Mask and Classification Usage
=============================

Photogrammetry is a technique that cannot reproduce altitude on water. This technique also has difficulties for moving elements or in shaded areas.

For this reason, it is possible to mask out areas or apply ad hoc processing to aid the matching stage.

Masks
-----

CARS can use a mask for each image in order to ignore some image regions (for instance water mask). This mask is taken into account during the whole 3D restitution process.

The masks have one band with 1 nbit and the non zeros values will be considered as invalid data and theses areas will not processed during the computation.

Please, see the section :ref:`convert_image_to_binary_image` to make a binary image.

The masks are resampled with the resampling application.

The masked values are not used in the matching process (sparse or dense matching method) to avoid mismatch and useless processing.

Further, the sparse matching estimation of the disparity range can be enhanced with mask using for the water area typicaly.

Classification
--------------

The considered classification image are multi-band raster with descriptions name for each banda and 1 nbit per band. See optional description option of the `Gdal band raster model <https://gdal.org/user/raster_data_model.html#raster-band>`_

Please, see the section :ref:`convert_image_to_binary_image` to make a multiband binary image with 1 bit per band.

All non-zeros values of the classification image will be considered as invalid data.

The classification can be used in each application by band name list selection parameter. See application ``classification`` parameter :ref:`configuration`..
