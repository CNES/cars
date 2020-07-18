set(DOCUMENTATION "OTB remote module for CARS")

# OTB_module() defines the module dependencies in ExternalTemplate
# ExternalTemplate depends on OTBCommon and OTBApplicationEngine
# The testing module in ExternalTemplate depends on OTBTestKernel
# and OTBCommandLine

# define the dependencies of the include module and the tests
otb_module(CARS
  DEPENDS
    OTBCommon
    OTBApplicationEngine
    OTBDescriptors
  TEST_DEPENDS
    OTBTestKernel
    OTBCommandLine
  DESCRIPTION
    "${DOCUMENTATION}"
)
