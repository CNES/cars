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
#include "otbOptiDisparityMapTo3DFilter.h"
#include "itkImageScanlineIterator.h"

namespace otb
{

namespace Wrapper
{

class EpipolarTriangulation : public otb::Wrapper::Application
{
public:
  typedef EpipolarTriangulation Self;
  typedef itk::SmartPointer<Self> Pointer; 

  itkNewMacro(Self);
  itkTypeMacro(EpipolarTriangulation, otb::Wrapper::Application);

  using FilterType = otb::DisparityMapTo3DFilter<FloatImageType,DoubleVectorImageType,FloatVectorImageType,Int16ImageType>;

  using RSTransformType = typename FilterType::RSTransformType;

private:
  void DoInit()
  {
    SetName("EpipolarTriangulation");
    SetDescription("Epipolar triangulation Algorithm");

    SetDocLimitations("None");
    SetDocAuthors("OTB-Team");
    SetDocSeeAlso("[1] StereoRectificationGridGenerator\n"
      "[2] GridBasedImageResampling");

    AddDocTag(Tags::Stereo);

    AddParameter(ParameterType_Choice,"mode","Triangulation mode");
    SetParameterDescription("mode","Triangulation mode (sift or disparity");
    
    AddChoice("mode.disp","Disparity");

    AddParameter(ParameterType_InputImage,"mode.disp.indisp","Horizontal disparity map");
    AddParameter(ParameterType_InputImage,"mode.disp.inmask","Mask of the disparity map if existing");
    MandatoryOff("mode.disp.inmask");

    AddChoice("mode.sift","Sift");
    AddParameter(ParameterType_InputImage,"mode.sift.inmatches","Matches buffer");

    AddParameter(ParameterType_InputImage,"leftgrid","Left epipolar grid");

    AddParameter(ParameterType_InputImage,"leftimage","Left image");
    AddParameter(ParameterType_InputImage,"rightimage","Right image");

    AddParameter(ParameterType_InputImage,"rightgrid","Left epipolar grid");

    AddParameter(ParameterType_Float,"leftminelev", "Minimum elevation for left LOS generation");
    AddParameter(ParameterType_Float,"leftmaxelev", "Maximum elevation for left LOS generation");

    AddParameter(ParameterType_Float,"rightminelev", "Minimum elevation for right LOS generation");
    AddParameter(ParameterType_Float,"rightmaxelev", "Maximum elevation for right LOS generation");

    AddParameter(ParameterType_OutputImage, "out", "The output 3D map, where each pixel is (lat,lon,elev)");
  }

  void DoUpdateParameters()
  {
  }

  void DoExecute()
  {  
    // Min and max elevation for LOS generation
    double leftMinElev = GetParameterFloat("leftminelev");
    double leftMaxElev = GetParameterFloat("leftmaxelev");
    double rightMinElev = GetParameterFloat("rightminelev");
    double rightMaxElev = GetParameterFloat("rightmaxelev");



    if(GetParameterString("mode") == "disp")
      {
      auto triangulator = FilterType::New();

      triangulator->SetHorizontalDisparityMapInput(GetParameterFloatImage("mode.disp.indisp"));
      triangulator->SetLeftEpipolarGridInput(GetParameterFloatVectorImage("leftgrid"));
      triangulator->SetRightEpipolarGridInput(GetParameterFloatVectorImage("rightgrid"));
      triangulator->SetLeftKeywordList(GetParameterFloatVectorImage("leftimage")->GetImageKeywordlist());
      triangulator->SetRightKeywordList(GetParameterFloatVectorImage("rightimage")->GetImageKeywordlist());
      triangulator->SetLeftMinimumElevation(leftMinElev);
      triangulator->SetLeftMaximumElevation(leftMaxElev);
      triangulator->SetRightMinimumElevation(rightMinElev);
      triangulator->SetRightMaximumElevation(rightMaxElev);



      if(IsParameterEnabled("mode.disp.inmask"))
        {
        triangulator->SetDisparityMaskInput(GetParameterInt16Image("mode.disp.inmask"));
        }
      
      SetParameterOutputImage("out",triangulator->GetOutput());
      
      RegisterPipeline();
      }
    else // mode  == sift
      {
      auto matches = GetParameterFloatImage("mode.sift.inmatches");
      matches->Update();

      auto outputRegion = matches->GetLargestPossibleRegion();
      auto outputSize = outputRegion.GetSize();
      outputSize[0] = 1;
      outputRegion.SetSize(outputSize);

      auto output = DoubleVectorImageType::New();
      DoubleVectorImageType::PixelType outputPoint(3);
      outputPoint.Fill(0);
      output->SetNumberOfComponentsPerPixel(3);
      output->SetRegions(outputRegion);
      output->Allocate();
      output->FillBuffer(outputPoint);


      itk::ImageScanlineConstIterator<FloatImageType> inIt(matches, matches->GetLargestPossibleRegion());
      itk::ImageScanlineIterator<DoubleVectorImageType> outIt(output,outputRegion);
      
      auto left_grid = GetParameterFloatVectorImage("leftgrid");
      auto right_grid = GetParameterFloatVectorImage("rightgrid");
      left_grid->Update();
      right_grid->Update();

      // Instantiate transforms
      auto leftToGroundTransform = RSTransformType::New();
      auto rightToGroundTransform = RSTransformType::New();
      
      leftToGroundTransform->SetInputKeywordList(GetParameterFloatVectorImage("leftimage")->GetImageKeywordlist());
      rightToGroundTransform->SetInputKeywordList(GetParameterFloatVectorImage("rightimage")->GetImageKeywordlist());

      leftToGroundTransform->InstantiateTransform();
      rightToGroundTransform->InstantiateTransform();

      inIt.GoToBegin();
      outIt.GoToBegin();

      while(!inIt.IsAtEnd() && !outIt.IsAtEnd())
        {
        
        DoubleImageType::PointType p1_epi, p2_epi;
        p1_epi[0] = inIt.Get();
        ++inIt;
        p1_epi[1] = inIt.Get();
        ++inIt;
        p2_epi[0] = inIt.Get();
        ++inIt;
        p2_epi[1] = inIt.Get();
        
        auto p1_sensor = otb::cars_details::InvertStereoRectificationGrid(p1_epi,left_grid);
        auto p2_sensor = otb::cars_details::InvertStereoRectificationGrid(p2_epi,right_grid);
        
        auto los1 = otb::cars_details::ComputeLineOfSight(p1_sensor,leftToGroundTransform.GetPointer(), leftMinElev, leftMaxElev);
        auto los2 = otb::cars_details::ComputeLineOfSight(p2_sensor,rightToGroundTransform.GetPointer(), rightMinElev, rightMaxElev);
        
        auto wgs84point = otb::cars_details::ToWGS84(otb::cars_details::Intersect<itk::Point<double,3>>(los1,los2));
        
        outputPoint[0] = wgs84point[0];
        outputPoint[1] = wgs84point[1];
        outputPoint[2] = wgs84point[2];

        outIt.Set(outputPoint);
        ++outIt;
        
        inIt.NextLine();
        }

      SetParameterOutputImage("out",output.GetPointer());
      }
  }
};

}
}

OTB_APPLICATION_EXPORT(otb::Wrapper::EpipolarTriangulation)
