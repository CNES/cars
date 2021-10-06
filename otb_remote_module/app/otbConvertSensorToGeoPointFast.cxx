/*
 * Copyright (c) 2020 Centre National d'Etudes Spatiales (CNES).
 *
 * This file is part of CARS
 * (see https://github.com/CNES/cars).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "otbWrapperApplication.h"
#include "otbWrapperApplicationFactory.h"

// Elevation handler
#include "otbWrapperElevationParametersHandler.h"

#include "otbForwardSensorModel.h"
#include "otbCoordinateToName.h"

namespace otb
{
namespace Wrapper
{

class ConvertSensorToGeoPointFast : public Application
{
public:
  /** Standard class typedefs. */
  typedef ConvertSensorToGeoPointFast       Self;
  typedef Application                   Superclass;
  typedef itk::SmartPointer<Self>       Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

  /** Standard macro */
  itkNewMacro(Self);

  itkTypeMacro(ConvertSensorToGeoPointFast, otb::Application);

  /** Filters typedef with 2 dimensions X, Y */
  typedef otb::ForwardSensorModel<double, 2, 3> ModelTypeXY;
  typedef itk::Point<double, 2> PointTypeXY;

  /** Filters typedef for model, point, index with 3 dimensions X, Y, Z */
  typedef otb::ForwardSensorModel<double, 3, 3> ModelTypeXYZ;
  typedef itk::Point<double, 3> PointTypeXYZ;
  typedef itk::ContinuousIndex<double, 3> IndexTypeXYZ;


private:
  void DoInit() override
  {
    SetName("ConvertSensorToGeoPointFast");
    SetDescription("Sensor to geographic coordinates conversion.");

    // Documentation
    std::ostringstream oss;
    oss << "This Application converts a sensor point of an input image";
    oss << "to a geographic point using the Forward Sensor Model of the input image.";
    oss << "Works with (X,Y) or (X,Y,H) depending on H value.";
    oss << "In Case in 2D, H is automatically set with OTB Elevation mechanisms";
    SetDocLongDescription(oss.str());

    SetDocLimitations("None");
    SetDocAuthors("OTB-Team");
    SetDocSeeAlso("ConvertCartoToGeoPoint application, otbObtainUTMZoneFromGeoPoint");

    AddDocTag(Tags::Geometry);

    AddParameter(ParameterType_InputImage, "in", "Sensor image");
    SetParameterDescription("in", "Input sensor image.");

    AddParameter(ParameterType_Group, "input", "Point Coordinates");
    AddParameter(ParameterType_Float, "input.idx", "X value of desired point");
    SetParameterDescription("input.idx", "X coordinate of the point to transform.");
    SetDefaultParameterFloat("input.idx", 0.0);
    AddParameter(ParameterType_Float, "input.idy", "Y value of desired point");
    SetParameterDescription("input.idy", "Y coordinate of the point to transform.");
    SetDefaultParameterFloat("input.idy", 0.0);
    AddParameter(ParameterType_Float,"input.idz", "Z altitude value of desired point above geoid");
    SetParameterDescription("input.idz", "Z altitude value of desired point above geoid");
    MandatoryOff("input.idz");


    // Output with Output Role
    AddParameter(ParameterType_Group, "output", "Geographic Coordinates");
    AddParameter(ParameterType_Double, "output.idx", "Output Point Longitude");
    SetParameterDescription("output.idx", "Output point longitude coordinate.");
    AddParameter(ParameterType_Double, "output.idy", "Output Point Latitude");
    SetParameterDescription("output.idy", "Output point latitude coordinate.");
    AddParameter(ParameterType_Double, "output.idz", "Output Point altitude");
    SetParameterDescription("output.idz", "Output point altitude coordinate.");

    // Set the parameter role for the output parameters
    SetParameterRole("output.idx", Role_Output);
    SetParameterRole("output.idy", Role_Output);
    SetParameterRole("output.idz", Role_Output);

    // Build the Output Elevation Parameter for XY option
    ElevationParametersHandler::AddElevationParameters(this, "elevation");

    // Doc example parameter settings
    SetDocExampleParameterValue("in", "QB_TOULOUSE_MUL_Extract_500_500.tif");
    SetDocExampleParameterValue("input.idx", "200");
    SetDocExampleParameterValue("input.idy", "200");

    SetOfficialDocLink();
  }

  void DoUpdateParameters() override
  {
    // Clear and reset the DEM Handler
    otb::DEMHandler::Instance()->ClearDEMs();
    otb::Wrapper::ElevationParametersHandler::SetupDEMHandlerFromElevationParameters(this, "elevation");

  }

  void DoExecute() override
  {
    // Handle elevation automatically with geoid, srtm or default elevation
    // respectively : elevation.geoid, elevation.dem, elevation.default
    otb::DEMHandler::Instance()->ClearDEMs();
    // the following needs OSSIM >= 2.0
    //~ ossimGeoidManager::instance()->clear();

    otb::Wrapper::ElevationParametersHandler::\
        SetupDEMHandlerFromElevationParameters(this,"elevation");

    // Get Input image
    FloatVectorImageType::Pointer inImage = GetParameterImage("in"); //Image

    //Declare and Instantiate a 2D X,Y Point
    PointTypeXY pointXY;

    // Declare and Instantiate a X,Y ContinuousIndex
    itk::ContinuousIndex<double, 2> inIndex;
    inIndex[0] = GetParameterFloat("input.idx");
    inIndex[1] = GetParameterFloat("input.idy");

    // Convert X, Y coordinates with img origin and spacing information
    inImage->TransformContinuousIndexToPhysicalPoint(inIndex, pointXY);

    if ( IsParameterEnabled("input.idz") && HasValue("input.idz") )
    {

      otbAppLogINFO("ConvertSensorToGeoPointFast with X,Y,Z inputs");

      // Instantiate a ForwardSensor XYZ Model
      ModelTypeXYZ::Pointer model = ModelTypeXYZ::New();
      model->SetImageGeometry(inImage->GetImageKeywordlist());
      if (model->IsValidSensorModel() == false)
      {
        itkGenericExceptionMacro(<< "Unable to create a model");
      }


      // Declare a XYZ point and transform 2D point to 3D
      PointTypeXYZ pointXYZ;
      pointXYZ[0] = pointXY[0];
      pointXYZ[1] = pointXY[1];
      pointXYZ[2] = GetParameterFloat("input.idz");

      // Declare OutputPoint
      ModelTypeXYZ::OutputPointType outputPoint;

      // Conversion of the desired point from Sensor to Geo Point
      outputPoint = model->TransformPoint(pointXYZ);

      // Set the value computed
      SetParameterDouble("output.idx", outputPoint[0]);
      SetParameterDouble("output.idy", outputPoint[1]);
      SetParameterDouble("output.idz", outputPoint[2]);
    }
    else
    {
      otbAppLogINFO("ConvertSensorToGeoPointFast with X,Y inputs only");

      // Declare and Instantiate a ForwardSensor XY Model
      ModelTypeXY::Pointer model = ModelTypeXY::New();
      model->SetImageGeometry(inImage->GetImageKeywordlist());

      if (model->IsValidSensorModel() == false)
      {
        itkGenericExceptionMacro(<< "Unable to create a model");
      }

      // Declare OutputPoint
      ModelTypeXY::OutputPointType outputPoint;

      // Conversion of the desired point from Sensor to Geo Point
      outputPoint = model->TransformPoint(pointXY);

      // Set the value computed
      SetParameterDouble("output.idx", outputPoint[0]);
      SetParameterDouble("output.idy", outputPoint[1]);
      SetParameterDouble("output.idz", outputPoint[2]);

    }
  }
};
}
}

OTB_APPLICATION_EXPORT(otb::Wrapper::ConvertSensorToGeoPointFast)
