Mask and Classification Usage
=============================

| Photogrammetry is a technique that cannot reproduce altitude on water. This technique also has difficulties for shaded areas, cloudy areas or moving elements such as cars.
| For this reason, it is possible to mask out areas or apply ad hoc processing to aid the matching stage.


Mask
-----

| CARS can use a mask for each image in order to ignore some image regions (for instance water mask). This mask is taken into account during the whole 3D restitution process.
| The masks have one band with 1 nbit and the non zeros values will be considered as invalid data and theses areas will not be processed during the computation.
| Please, see the section :ref:`convert_image_to_binary_image` to make a binary image.

| The masks are resampled in epipolar geometry with the resampling application.
| The masked values are not taken into account in the matching process (sparse or dense matching method) to avoid mismatch and useless processing.
| Furthermore, the sparse matching estimation of the disparity range can be enhanced with mask using for the water area typicaly.

Classification
--------------

| The considered classification image are multi-band raster with descriptions name for each band and 1 nbit per band. See optional description option of the `Gdal band raster model <https://gdal.org/user/raster_data_model.html#raster-band>`_
| Please, see the section :ref:`convert_image_to_binary_image` to make a multiband binary image with 1 bit per band.

| All non-zeros values of the classification image will be considered as invalid data.
| The classification can be used in each application by band name list selection parameter. See application ``classification`` parameter :ref:`advanced configuration`.