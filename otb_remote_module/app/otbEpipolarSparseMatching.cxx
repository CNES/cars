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
#include "otbDisparityMapTo3DFilter.h"

#include "otbSiftFastImageFilter.h"
#include "otbVlfeatSiftImageFilter.h"
#include "otbImageToSIFTKeyPointSetFilter.h"
#include "otbImageToSURFKeyPointSetFilter.h"
#include "otbKeyPointSetsMatchingFilter.h"
#include "otbExtractROI.h"
#include "itkImageRegionIterator.h"
#include "itkExtractImageFilter.h"
#include "otbStopwatch.h"

#include <algorithm>
#include <deque>
#include <omp.h>

namespace otb
{

namespace Wrapper
{

class EpipolarSparseMatching : public otb::Wrapper::Application
{
public:
  typedef EpipolarSparseMatching Self;
  typedef itk::SmartPointer<Self> Pointer;

  itkNewMacro(Self);
  itkTypeMacro(EpipolarSparseMatching, otb::Wrapper::Application);

  using ExtractFilterType = ExtractROI<FloatImageType::PixelType, FloatImageType::PixelType>;
  using RealVectorType = FloatVectorImageType::PixelType;
  using MaskType = Int16ImageType::PixelType;
  using ExtractMaskFilterType = ExtractROI<Int16ImageType::PixelType, Int16ImageType::PixelType>;
  using PointSetType = itk::PointSet<RealVectorType,2>;
  //using SiftFilterType = ImageToSIFTKeyPointSetFilter<FloatImageType,PointSetType>;
  //using SiftFilterType = otb::SiftFastImageFilter<FloatImageType,PointSetType>;
  using SiftFilterType = otb::VlfeatSiftImageFilter<FloatImageType,PointSetType>;
  using SurfFilterType = ImageToSURFKeyPointSetFilter<FloatImageType,PointSetType>;
  using DistanceType = itk::Statistics::EuclideanDistanceMetric<RealVectorType>;
  using MatchingFilterType = KeyPointSetsMatchingFilter<PointSetType,DistanceType>;
  using ConstIteratorType = itk::ImageRegionConstIterator<Int16ImageType>;
  using PointSetIteratorType = PointSetType::PointsContainer::Iterator;
  using PointDataIteratorType = PointSetType::PointDataContainer::Iterator;

private:
  void DoInit()
  {
    SetName("EpipolarSparseMatching");
    SetDescription("Epipolar Sparse Matching Algorithm");

    SetDocLimitations("None");
    SetDocAuthors("OTB-Team");
    SetDocSeeAlso("[1] StereoRectificationGridGenerator\n"
                  "[2] GridBasedImageResampling");

    AddDocTag(Tags::Stereo);

    AddParameter(ParameterType_InputImage, "in1", "Input Image 1");
    SetParameterDescription("in1"," First input image");

    AddParameter(ParameterType_InputImage, "in2", "Input Image 2");
    SetParameterDescription("in2"," Second input image");

    AddParameter(ParameterType_InputImage, "inmask1", "Mask for input image 1");
    SetParameterDescription("inmask1"," Mask for first input image");
    MandatoryOff("inmask1"),

    AddParameter(ParameterType_InputImage, "inmask2", "Mask for input image 2");
    SetParameterDescription("inmask2"," Mask for second input image");
    MandatoryOff("inmask2"),

    AddParameter(ParameterType_Int,"maskvalue","Value to indicate a good pixel in mask");
    SetDefaultParameterInt("maskvalue",0);

    AddParameter(ParameterType_Choice,"algorithm","Keypoints detection algorithm");
    SetParameterDescription("algorithm","Choice of the detection algorithm to use");

    AddChoice("algorithm.sift","SIFT algorithm");
    AddChoice("algorithm.surf","SURF algorithm");

    AddParameter(ParameterType_Float,"matching", "Distance threshold for matching");
    SetParameterDescription("matching","The difference of Gaussian response threshold for matching.");
    SetMinimumParameterFloatValue("matching",0.0);
    SetDefaultParameterFloat("matching", 0.6);

    AddParameter(ParameterType_Int, "octaves", "Number of octave");
    SetParameterDescription("octaves", "Number of octave for SIFT commputation.");
    SetMinimumParameterIntValue("octaves", 1);
    SetDefaultParameterInt("octaves", 8);

    AddParameter(ParameterType_Int, "scales", "Number of scale per octave");
    SetParameterDescription("scales", "Number of scale per octave for SIFT commputation.");
    SetMinimumParameterIntValue("scales", 1);
    SetDefaultParameterInt("scales", 3);

    AddParameter(ParameterType_Float,"tdog","DoG threshold");
    SetParameterDescription("tdog","The difference of Gaussian response threshold for SIFT keypoints.");
    SetMinimumParameterFloatValue("tdog",0.0);
    SetDefaultParameterFloat("tdog", 0.0133333);

    AddParameter(ParameterType_Float,"tedge","Edge threshold");
    SetParameterDescription("tedge","The edge threshold for SIFT.");
    SetMinimumParameterFloatValue("tedge",0.0);
    SetDefaultParameterFloat("tedge",10.0);

    AddParameter(ParameterType_Float, "magnification", "Keypoint magnification factor.");
    SetParameterDescription("magnification", "Keypoint scale is multiplied by this factor.");
    SetMinimumParameterFloatValue("magnification", 0.0);
    SetDefaultParameterFloat("magnification", 3.0);

    AddParameter(ParameterType_Bool,"backmatching","Use back-matching to filter matches");
    SetParameterDescription("backmatching","If set to true, matches should be consistent in both ways");

    AddParameter(ParameterType_OutputImage, "out", "image of matches of shape (N,4)");

    AddParameter(ParameterType_Int,"nbmatches", "Number of matches found");
    SetParameterRole("nbmatches",Role_Output);
  }

  void DoUpdateParameters()
  {
  }


  PointSetType::Pointer ComputePointsSIFT(FloatImageType* in)
  {
    auto sift = SiftFilterType::New();
    auto n_octave = GetParameterInt("octaves");
    auto n_scale_per_octave = GetParameterInt("scales");
    auto dog_threshold = GetParameterFloat("tdog");
    auto edge_threshold = GetParameterFloat("tedge");
    auto magnification = GetParameterFloat("magnification");
    sift->SetInput(in);
    sift->SetNumberOfOctaves(n_octave);
    sift->SetNumberOfScalesPerOctave(n_scale_per_octave);
    sift->SetDoGThreshold(dog_threshold);
    sift->SetEdgeThreshold(edge_threshold);
    sift->SetMagnification(magnification);
    sift->Update();
    return sift->GetOutput();
  }

  PointSetType::Pointer ComputePointsSURF(FloatImageType* in)
  {
    auto surf = SurfFilterType::New();
    surf->SetInput(in);
    surf->Update();
    return surf->GetOutput();
  }


  PointSetType::Pointer ComputePoints(FloatImageType * in, Int16ImageType * mask, std::string algorithm, Int16ImageType::ValueType maskValue)
  {
    PointSetType::Pointer points;
    if (algorithm == "sift")
      {
      points = ComputePointsSIFT(in);
      }
    else
      {
      points = ComputePointsSURF(in);
      }
    PointSetType::Pointer keptPoints;
    if (mask)
      {
      keptPoints = PointSetType::New();
      PointSetIteratorType it = points->GetPoints()->Begin();
      PointDataIteratorType itData = points->GetPointData()->Begin();
      Int16ImageType::IndexType index;
      auto i = 0;
      for (;
           it != points->GetPoints()->End()
           && itData != points->GetPointData()->End();
           ++it, ++itData)
        {
           if (mask->TransformPhysicalPointToIndex(it.Value(), index))
             {
             if (mask->GetPixel(index) == maskValue)
               {
               keptPoints->SetPoint(i,it.Value());
               keptPoints->SetPointData(i,itData.Value());
               ++i;
               }
             }
        }
      }
    else
      {
      keptPoints = points;
      }
    return keptPoints;
  }

  bool IsNotMasked(Int16ImageType::Pointer mask, Int16ImageType::ValueType maskValue)
  {
    Int16ImageType::ValueType maskvalue = GetParameterInt("maskvalue");
    ConstIteratorType it(mask, mask->GetLargestPossibleRegion());
    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
      {
        if (it.Get() == maskvalue) return true;
      }
    return false;
  }

  void DoExecute()
  {
    #pragma omp
    auto threshold = GetParameterFloat("matching");
    auto algorithm = GetParameterString("algorithm");
    bool backmatching = GetParameterInt("backmatching");
    Int16ImageType::ValueType maskValue = GetParameterInt("maskvalue");
    auto nb_matches = 0UL;


    // Update left image and mask and discard left image if fully masked
    FloatImageType::Pointer leftImage = GetParameterFloatImage("in1");
    leftImage->Update();
    Int16ImageType::Pointer leftMask = nullptr;
    if(IsParameterEnabled("inmask1"))
      {
      leftMask =  GetParameterInt16Image("inmask1");
      leftMask->Update();
      if (!IsNotMasked(leftMask,maskValue))
        {
        // No valid data in mask, so skip image
        leftImage = nullptr;
        }
      }

    // Update right image and mask and discard right image if fully masked
    FloatImageType::Pointer rightImage = GetParameterFloatImage("in2");
    rightImage->Update();
    Int16ImageType::Pointer rightMask = nullptr;
    if(IsParameterEnabled("inmask2"))
      {
      rightMask = GetParameterInt16Image("inmask2");
      rightMask->Update();
      if (!IsNotMasked(rightMask,maskValue))
        {
        // No valid data in mask, so skip image
        rightImage = nullptr;
        }
      }

      // If both images are valid
      if (leftImage && rightImage)
        {
        // Compute keypoints

        otb::Stopwatch clock;
        clock.Start();
        PointSetType::Pointer leftPoints = ComputePoints(leftImage, leftMask, algorithm, maskValue);
        PointSetType::Pointer rightPoints = ComputePoints(rightImage, rightMask, algorithm, maskValue);
        clock.Stop();

        otbAppLogINFO(<<"Points detection: "<<clock.GetElapsedMilliseconds()<<" ms, "<<leftPoints->GetNumberOfPoints()<<" x "<<rightPoints->GetNumberOfPoints()<<" points");

        // If there are points to match
	// need minimum two right points to find the second nearest neighbor
        if(leftPoints->GetNumberOfPoints() > 0
           && rightPoints->GetNumberOfPoints() > 1)
          {
          // Compute matching
          auto matchingFilter = MatchingFilterType::New();
          matchingFilter->SetDistanceThreshold(threshold);
          matchingFilter->SetUseBackMatching(backmatching);

          matchingFilter->SetInput1(leftPoints);
          matchingFilter->SetInput2(rightPoints);

          try
            {
            clock.Reset();
            clock.Start();
            matchingFilter->Update();
            clock.Stop();

            nb_matches = matchingFilter->GetOutput()->Size();
            otbAppLogINFO(<<"Points matching: "<<clock.GetElapsedMilliseconds()<<" ms, "<<nb_matches<<" matches");


            if(nb_matches > 0)
              {
              auto outputArray = FloatImageType::New();
              FloatImageType::RegionType outputArrayRegion;
              FloatImageType::SizeType outputArraySize = {{4,nb_matches}};
              outputArrayRegion.SetSize(outputArraySize);

              outputArray->SetRegions(outputArrayRegion);
              outputArray->Allocate();

              itk::ImageRegionIterator<FloatImageType> it(outputArray,outputArrayRegion);

              for(auto mIt = matchingFilter->GetOutput()->Begin(); mIt != matchingFilter->GetOutput()->End(); ++mIt)
                  {
                  it.Set(mIt.Get()->GetPoint1()[0]);
                  ++it;
                  it.Set(mIt.Get()->GetPoint1()[1]);
                  ++it;
                  it.Set(mIt.Get()->GetPoint2()[0]);
                  ++it;
                  it.Set(mIt.Get()->GetPoint2()[1]);
                  ++it;
                  }

              SetParameterOutputImage("out",outputArray.GetPointer());
              }
            }
          catch(itk::ExceptionObject & err)
            {
            otbAppLogWARNING("Warning: tile ignored because of the following error "<<err);
            }
          }
        }

      if(nb_matches == 0)
        {
        // Dummy output
        auto outputArray = FloatImageType::New();
        FloatImageType::RegionType outputArrayRegion;
        FloatVectorImageType::SizeType outputArraySize = {{4,1}};
        outputArrayRegion.SetSize(outputArraySize);
        outputArray->SetRegions(outputArrayRegion);
        outputArray->Allocate();
        outputArray->FillBuffer(0);
        SetParameterOutputImage("out",outputArray.GetPointer());
        }

      SetParameterInt("nbmatches",nb_matches);
  }
};

}
}

OTB_APPLICATION_EXPORT(otb::Wrapper::EpipolarSparseMatching)
