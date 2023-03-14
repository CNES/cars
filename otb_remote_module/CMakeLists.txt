cmake_minimum_required(VERSION 2.8.9)
project(CARS)

if(NOT OTB_SOURCE_DIR)
  # Handle the compilation outside OTB source tree: find where OTB is built/installed
  find_package(OTB REQUIRED)
  list(APPEND CMAKE_MODULE_PATH ${OTB_CMAKE_DIR})
  include(UseOTB)

  # This script will process the module sources
  include(OTBModuleExternal)

else()
  otb_module_impl()
endif()

