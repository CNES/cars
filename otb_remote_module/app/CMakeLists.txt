OTB_CREATE_APPLICATION(NAME EpipolarTriangulation
                       SOURCES otbEpipolarTriangulation.cxx
                       LINK_LIBRARIES ${${otb-module}_LIBRARIES})

OTB_CREATE_APPLICATION(NAME ConvertSensorToGeoPointFast
                       SOURCES otbConvertSensorToGeoPointFast.cxx
                       LINK_LIBRARIES ${${otb-module}_LIBRARIES})



