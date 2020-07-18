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

  /** Filters typedef */
  typedef otb::ForwardSensorModel<double> ModelType;
  typedef itk::Point<double, 2> PointType;

private:
  void DoInit() override
  {
    SetName("ConvertSensorToGeoPointFast");
    SetDescription("Sensor to geographic coordinates conversion.");

    // Documentation
    SetDocLongDescription(
        "This Application converts a sensor point of an input image to a geographic point using the Forward Sensor Model of the input image.");
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

    // Output with Output Role
    AddParameter(ParameterType_Group, "output", "Geographic Coordinates");
    AddParameter(ParameterType_Float, "output.idx", "Output Point Longitude");
    SetParameterDescription("output.idx", "Output point longitude coordinate.");
    AddParameter(ParameterType_Float, "output.idy", "Output Point Latitude");
    SetParameterDescription("output.idy", "Output point latitude coordinate.");

    AddParameter(ParameterType_String, "output.town", "Main town near the coordinates computed");
    SetParameterDescription("output.town", "Nearest main town of the computed geographic point.");
    AddParameter(ParameterType_String, "output.country", "Country of the image");
    SetParameterDescription("output.country", "Country of the input image");

    // Set the parameter role for the output parameters
    SetParameterRole("output.idx", Role_Output);
    SetParameterRole("output.idy", Role_Output);

    ElevationParametersHandler::AddElevationParameters(this, "elevation");

    // Doc example parameter settings
    SetDocExampleParameterValue("in", "QB_TOULOUSE_MUL_Extract_500_500.tif");
    SetDocExampleParameterValue("input.idx", "200");
    SetDocExampleParameterValue("input.idy", "200");

    SetOfficialDocLink();
  }

  void DoUpdateParameters() override
  {
  }

  void DoExecute() override
  {
    // Handle elevation
    otb::Wrapper::ElevationParametersHandler::SetupDEMHandlerFromElevationParameters(this, "elevation");
  
    // Get input Image
    FloatVectorImageType::Pointer inImage = GetParameterImage("in");

    // Instantiate a ForwardSensor Model
    ModelType::Pointer model = ModelType::New();
    model->SetImageGeometry(inImage->GetImageKeywordlist());
    if (model->IsValidSensorModel() == false)
    {
      itkGenericExceptionMacro(<< "Unable to create a model");
    }

    // Convert the desired point
    itk::ContinuousIndex<double, 2> inIndex;
    PointType point;

    inIndex[0] = GetParameterFloat("input.idx");
    inIndex[1] = GetParameterFloat("input.idy");
    inImage->TransformContinuousIndexToPhysicalPoint(inIndex, point);

    ModelType::OutputPointType outputPoint;
    outputPoint = model->TransformPoint(point);

    // Set the value computed
    SetParameterFloat("output.idx", outputPoint[0]);
    SetParameterFloat("output.idy", outputPoint[1]);
  }
};
}
}

OTB_APPLICATION_EXPORT(otb::Wrapper::ConvertSensorToGeoPointFast)
