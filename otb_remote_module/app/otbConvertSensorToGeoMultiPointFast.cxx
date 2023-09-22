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
#include "itkImageRegionIterator.h"
#include "otbForwardSensorModel.h"
#include "otbCoordinateToName.h"
#include <vector>

namespace otb
{
  namespace Wrapper
  {

    class ConvertSensorToGeoMultiPointFast : public Application
    {
    public:
      /** Standard class typedefs. */
      typedef ConvertSensorToGeoMultiPointFast Self;
      typedef Application Superclass;
      typedef itk::SmartPointer<Self> Pointer;
      typedef itk::SmartPointer<const Self> ConstPointer;

      typedef otb::Image<double, 2> ImageType;
      typedef itk::ImageRegionIterator<ImageType> IteratorType;

      /** Standard macro */
      itkNewMacro(Self);

      itkTypeMacro(ConvertSensorToGeoMultiPointFast, otb::Application);

      /** Filters typedef with 2 dimensions X, Y */
      typedef otb::ForwardSensorModel<double, 2, 3> ModelTypeXY;
      typedef itk::Point<double, 2> PointTypeXY;

      /** Filters typedef for model, point, index with 3 dimensions X, Y, Z */
      typedef otb::ForwardSensorModel<double, 3, 3> ModelTypeXYZ;
      typedef itk::Point<double, 3> PointTypeXYZ;

    private:
      void DoInit() override
      {
        SetName("ConvertSensorToGeoMultiPointFast");
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

        AddParameter(ParameterType_Group, "input", "Points Coordinates List");
        AddParameter(ParameterType_StringList, "input.idx", "X values of desired points list");
        SetParameterDescription("input.idx", "X coordinates of the points list to transform.");
        AddParameter(ParameterType_StringList, "input.idy", "Y values of desired points list");
        SetParameterDescription("input.idy", "Y coordinates of the points list to transform.");
        AddParameter(ParameterType_StringList, "input.idz", "Z altitudes values of desired points list above geoid");
        SetParameterDescription("input.idz", "Z altitudes values of desired points list above geoid");

        MandatoryOff("input.idz");

        // Output with Output Role
        AddParameter(ParameterType_Group, "output", "Geographic Coordinates");
        AddParameter(ParameterType_OutputImage, "output.all", "Output image as lon lat alt data array");
        SetParameterDescription("output.all", "Output image data Longitude, latitude, alt coordinate.");

        // Set the parameter role for the output parameters
        SetParameterRole("output.all", Role_Output);

        // Build the Output Elevation Parameter for XY option
        ElevationParametersHandler::AddElevationParameters(this, "elevation");

        // Doc example parameter settings
        SetDocExampleParameterValue("in", "QB_TOULOUSE_MUL_Extract_500_500.tif");
        SetDocExampleParameterValue("input.idx", "[200,300]");
        SetDocExampleParameterValue("input.idy", "[200,300]");

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

        otb::Wrapper::ElevationParametersHandler::
            SetupDEMHandlerFromElevationParameters(this, "elevation");

        // Get Input image
        FloatVectorImageType::Pointer inImage = GetParameterImage("in"); // Image

        // Declare and Instantiate a 2D X,Y Point
        PointTypeXY pointXY;
        std::vector<std::string> indexListX = GetParameterStringList("input.idx");
        std::vector<std::string> indexListY = GetParameterStringList("input.idy");
        std::vector<std::string> indexListZ = GetParameterStringList("input.idz");

        // Declare region to define size of lon lat alt array to get on image output format
        ImageType::RegionType inputRegion;
        ImageType::RegionType::SizeType size;
        ImageType::RegionType outputRegion;
        ImageType::RegionType::IndexType outputStart;

        outputStart[0] = 0;
        outputStart[1] = 0;
        // tot: check inversion size
        size[0] = 3;
        size[1] = indexListX.size();

        outputRegion.SetSize(size);
        outputRegion.SetIndex(outputStart);
        ImageType::Pointer outputImage = ImageType::New();
        outputImage->SetRegions(outputRegion);
        const double spacing[2] = {1, 1};
        const double origin[2] = {0, 0};
        outputImage->SetSpacing(spacing);
        outputImage->SetOrigin(origin);
        outputImage->Allocate();
        IteratorType outputIt(outputImage, outputRegion);
        outputIt.GoToBegin();

        bool hasZvalue = false;
        if (IsParameterEnabled("input.idz") && HasValue("input.idz"))
        {
          hasZvalue = true;
        }

        int k = 0;
        for (unsigned int i = 0; i < indexListX.size(); i++)
        {
          double valx = boost::lexical_cast<float>(indexListX[i]);
          double valy = boost::lexical_cast<float>(indexListY[i]);
          double valz = 0;
          if (hasZvalue)
          {
            valz = boost::lexical_cast<float>(indexListZ[i]);
          }
          // Declare and Instantiate a X,Y ContinuousIndex
          itk::ContinuousIndex<double, 2> inIndex;
          inIndex[0] = valx;
          inIndex[1] = valy;

          // Convert X, Y coordinates with img origin and spacing information
          inImage->TransformContinuousIndexToPhysicalPoint(inIndex, pointXY);

          if (hasZvalue)
          {
            otbAppLogINFO("ConvertSensorToGeoMultiPointFast with X,Y,Z inputs");

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
            pointXYZ[2] = valz;

            // Declare OutputPoint
            ModelTypeXYZ::OutputPointType outputPoint;

            // Conversion of the desired point from Sensor to Geo Point
            outputPoint = model->TransformPoint(pointXYZ);

            // Set the value computed
            outputIt.Set(outputPoint[1]);
            ++outputIt;
            outputIt.Set(outputPoint[0]);
            ++outputIt;
            outputIt.Set(outputPoint[2]);
            ++outputIt;
          }
          else
          {
            otbAppLogINFO("ConvertSensorToGeoMultiPointFast with X,Y inputs only");

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
            // ImageType point;
            outputIt.Set(outputPoint[1]);
            ++outputIt;
            outputIt.Set(outputPoint[0]);
            ++outputIt;
            outputIt.Set(outputPoint[2]);
            ++outputIt;
          }
        }
        SetParameterOutputImage<ImageType>("output.all", outputImage);
      }
    };
  }
}

OTB_APPLICATION_EXPORT(otb::Wrapper::ConvertSensorToGeoMultiPointFast)
