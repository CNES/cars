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
#include "otbInverseSensorModel.h"
#include "otbGenericRSTransform.h"

#include "otbCoordinateToName.h"

namespace otb
{
namespace Wrapper
{

class LocalizeInverse : public Application
{
public:
  /** Standard class typedefs. */
  typedef LocalizeInverse       Self;
  typedef Application                   Superclass;
  typedef itk::SmartPointer<Self>       Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

  /** Standard macro */
  itkNewMacro(Self);

  itkTypeMacro(LocalizeInverse, otb::Application);

  /** Filters typedef */
  typedef otb::ForwardSensorModel<double,3,3> ModelType;
  typedef otb::ForwardSensorModel<double, 2, 3> ModelType2D;
  typedef otb::InverseSensorModel<double,3,3> InverseModelType;
  typedef otb::GenericRSTransform<double,3,3>       RSTransformType;
  typedef itk::Point<double, 2> PointType;
  typedef itk::Point<double, 3> Point3DType;
  typedef itk::Point<double, 3> PointTypeXYZ;
  typedef otb::Image<double, 2> ImageType;
  typedef itk::ImageRegionIterator<ImageType> IteratorType;

private:
  void DoInit() override
  {
    SetName("LocalizeInverse");
    SetDescription("Localize inverse.");

    // Documentation
    SetDocLongDescription(
        "This Application converts a sensor point of an input image to a geographic point using the Forward Sensor Model of the input image.");
    SetDocLimitations("None");
    SetDocAuthors("OTB-Team");
    SetDocSeeAlso("ConvertCartoToGeoPoint application, otbObtainUTMZoneFromGeoPoint");

    AddDocTag(Tags::Geometry);

    AddParameter(ParameterType_InputImage, "in", "Sensor image");
    SetParameterDescription("in", "Input sensor image.");

    AddParameter(ParameterType_Choice,"coordtype","otb index or physical point");
    SetParameterDescription("coordtype","index or physical point  (index/ physical)");

    AddChoice("coordtype.physical","physical");
    AddChoice("coordtype.index","index");

    // Output with Output Role
    AddParameter(ParameterType_Group, "output", "Geographic Coordinates");
    AddParameter(ParameterType_OutputImage, "output.all", "Output image as lon lat alt data array");
    SetParameterDescription("output.all", "Output image data Longitude, latitude, alt coordinate.");

    // Set the parameter role for the output parameters
    SetParameterRole("output.all", Role_Output);



    AddParameter(ParameterType_Group, "input", "Point Coordinates");
    AddParameter(ParameterType_StringList, "input.idx", "X value of desired point");
    SetParameterDescription("input.idx", "X coordinate of the point to transform.");
    AddParameter(ParameterType_StringList, "input.idy", "Y value of desired point");
    SetParameterDescription("input.idy", "Y coordinate of the point to transform.");
    AddParameter(ParameterType_StringList, "input.idz", "Z value of desired point");
    SetParameterDescription("input.idz", "Z coordinate of the point to transform.");

    // Output with Output Role
    AddParameter(ParameterType_Group, "output", "Geographic Coordinates");
    AddParameter(ParameterType_OutputImage, "output.all", "Output image as lon lat alt data array");
    SetParameterDescription("output.all", "Output image data Longitude, latitude, alt coordinate.");

    // Set the parameter role for the output parameters
    AddParameter(ParameterType_Group, "output", "Geographic Coordinates");
    AddParameter(ParameterType_OutputImage, "output.all", "Output image as lon lat alt data array");
    SetParameterDescription("output.all", "Output image data Longitude, latitude, alt coordinate.");

    // Set the parameter role for the output parameters
    SetParameterRole("output.all", Role_Output);

    ElevationParametersHandler::AddElevationParameters(this, "elevation");


    // Doc example parameter settings
    SetDocExampleParameterValue("in", "QB_TOULOUSE_MUL_Extract_500_500.tif");
    SetDocExampleParameterValue("input.idx", "[200,300]");
    SetDocExampleParameterValue("input.idy", "[200,300]");

    SetOfficialDocLink();
  }

  void DoUpdateParameters() override
  {
  }

  void DoExecute() override
  {

    // Handle elevation automatically with geoid, srtm or default elevation
    // respectively : elevation.geoid, elevation.dem, elevation.default
    otb::DEMHandler::Instance()->ClearDEMs();
    // the following needs OSSIM >= 2.0
    //~ ossimGeoidManager::instance()->clear();
    // Handle elevation
    otb::Wrapper::ElevationParametersHandler::SetupDEMHandlerFromElevationParameters(this, "elevation");
  
    // Get input Image
    FloatVectorImageType::Pointer inImage = GetParameterImage("in");

    // Instantiate a ForwardSensor Model
    ModelType::Pointer model = ModelType::New();
    InverseModelType::Pointer inverseModel = InverseModelType::New();


    model->SetImageGeometry(inImage->GetImageKeywordlist());
    if (model->IsValidSensorModel() == false)
    {
      itkGenericExceptionMacro(<< "Unable to create a model");
    }
    inverseModel->SetImageGeometry(inImage->GetImageKeywordlist());

    RSTransformType::Pointer RSmodel = RSTransformType::New();

    RSmodel->SetInputKeywordList(inImage->GetImageKeywordlist());
    RSmodel->InstantiateTransform();


    // Point3DType point3D;
    // point3D[0] = GetParameterDouble("input.idx");
    // point3D[1] = GetParameterDouble("input.idy");
    // point3D[2] = GetParameterDouble("input.idz");
    // Declare and Instantiate a 2D X,Y Point
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
 	

    int k = 0;
    for (unsigned int i = 0; i < indexListX.size(); i++)
    {
      double valx = boost::lexical_cast<double>(indexListX[i]);
      double valy = boost::lexical_cast<double>(indexListY[i]);
      double valz = boost::lexical_cast<double>(indexListZ[i]);

      Point3DType point3D;
      point3D[0] = valx;
      point3D[1] = valy;
      point3D[2] = valz;

      ModelType::OutputPointType outputPoint;
      InverseModelType::OutputPointType inverseOutputPoint;
      outputPoint = inverseModel->TransformPoint(point3D);
    
      if(GetParameterString("coordtype") == "index")
      {	
        PointType point;
        point[0] = outputPoint[0];
        point[1] = outputPoint[1];
        itk::ContinuousIndex<double, 2> outIndex;
        inImage->TransformPhysicalPointToContinuousIndex(point,outIndex);
        outputPoint[0] = outIndex[0];
        outputPoint[1] = outIndex[1];
      }
      // Set the value computed
      // ImageType point;
      outputIt.Set(outputPoint[0]);
      ++outputIt;
      outputIt.Set(outputPoint[1]);
      ++outputIt;
      outputIt.Set(outputPoint[2]);
      ++outputIt;

    }

    SetParameterOutputImage<ImageType>("output.all", outputImage);
  }
};
}
}

OTB_APPLICATION_EXPORT(otb::Wrapper::LocalizeInverse)
