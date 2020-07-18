OTB_CREATE_APPLICATION(NAME EpipolarTriangulation
                       SOURCES otbEpipolarTriangulation.cxx
                       LINK_LIBRARIES ${${otb-module}_LIBRARIES})

OTB_CREATE_APPLICATION(NAME EpipolarSparseMatching
                       SOURCES otbEpipolarSparseMatching.cxx
                       LINK_LIBRARIES ${${otb-module}_LIBRARIES} ${VLFEAT_LIBRARY})

OTB_CREATE_APPLICATION(NAME BuildMask
                       SOURCES otbBuildMask.cxx
                       LINK_LIBRARIES ${${otb-module}_LIBRARIES})

OTB_CREATE_APPLICATION(NAME DEMReader
                       SOURCES otbDEMReader.cxx
                       LINK_LIBRARIES ${${otb-module}_LIBRARIES})

OTB_CREATE_APPLICATION(NAME ConvertSensorToGeoPointFast
                       SOURCES otbConvertSensorToGeoPointFast.cxx
                       LINK_LIBRARIES ${${otb-module}_LIBRARIES})



