OTB_CREATE_APPLICATION(NAME EpipolarTriangulation
                       SOURCES otbEpipolarTriangulation.cxx
                       LINK_LIBRARIES ${${otb-module}_LIBRARIES})

OTB_CREATE_APPLICATION(NAME ConvertSensorToGeoMultiPointFast
                       SOURCES otbConvertSensorToGeoMultiPointFast.cxx
                       LINK_LIBRARIES ${${otb-module}_LIBRARIES})


