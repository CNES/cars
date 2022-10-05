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

#ifndef otbOptiDisparityMapTo3DFilter_hxx
#define otbOptiDisparityMapTo3DFilter_hxx

#include "otbOptiDisparityMapTo3DFilter.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkImageRegionIterator.h"


namespace otb
{

template <class TDisparityImage, class TOutputImage,
class TEpipolarGridImage, class TMaskImage>
DisparityMapTo3DFilter<TDisparityImage,TOutputImage,TEpipolarGridImage,TMaskImage>
::DisparityMapTo3DFilter()
{
  m_LeftMinimumElevation =  0;   // meters
  m_LeftMaximumElevation =  300; // meters
  m_RightMinimumElevation =  0;   // meters
  m_RightMaximumElevation =  300; // meters

  // Set the number of inputs
  this->SetNumberOfRequiredInputs(5);
  this->SetNumberOfRequiredInputs(1);

  // Set the outputs
  this->SetNumberOfRequiredOutputs(1);
  this->SetNthOutput(0,TOutputImage::New());
}

template <class TDisparityImage, class TOutputImage,
class TEpipolarGridImage, class TMaskImage>
DisparityMapTo3DFilter<TDisparityImage,TOutputImage,TEpipolarGridImage,TMaskImage>
::~DisparityMapTo3DFilter()
{}

template <class TDisparityImage, class TOutputImage,
class TEpipolarGridImage, class TMaskImage>
void
DisparityMapTo3DFilter<TDisparityImage,TOutputImage,TEpipolarGridImage,TMaskImage>
::SetHorizontalDisparityMapInput( const TDisparityImage * hmap)
{
  // Process object is not const-correct so the const casting is required.
  this->SetNthInput(0, const_cast<TDisparityImage *>( hmap ));
}

template <class TDisparityImage, class TOutputImage,
class TEpipolarGridImage, class TMaskImage>
void
DisparityMapTo3DFilter<TDisparityImage,TOutputImage,TEpipolarGridImage,TMaskImage>
::SetVerticalDisparityMapInput( const TDisparityImage * vmap)
{
  // Process object is not const-correct so the const casting is required.
  this->SetNthInput(1, const_cast<TDisparityImage *>( vmap ));
}

template <class TDisparityImage, class TOutputImage,
class TEpipolarGridImage, class TMaskImage>
void
DisparityMapTo3DFilter<TDisparityImage,TOutputImage,TEpipolarGridImage,TMaskImage>
::SetLeftEpipolarGridInput( const TEpipolarGridImage * grid)
{
  // Process object is not const-correct so the const casting is required.
  this->SetNthInput(2, const_cast<TEpipolarGridImage *>( grid ));
}

template <class TDisparityImage, class TOutputImage,
class TEpipolarGridImage, class TMaskImage>
void
DisparityMapTo3DFilter<TDisparityImage,TOutputImage,TEpipolarGridImage,TMaskImage>
::SetRightEpipolarGridInput( const TEpipolarGridImage * grid)
{
  // Process object is not const-correct so the const casting is required.
  this->SetNthInput(3, const_cast<TEpipolarGridImage *>( grid ));
}

template <class TDisparityImage, class TOutputImage,
class TEpipolarGridImage, class TMaskImage>
void
DisparityMapTo3DFilter<TDisparityImage,TOutputImage,TEpipolarGridImage,TMaskImage>
::SetDisparityMaskInput( const TMaskImage * mask)
{
  // Process object is not const-correct so the const casting is required.
  this->SetNthInput(4, const_cast<TMaskImage *>( mask ));
}

template <class TDisparityImage, class TOutputImage,
class TEpipolarGridImage, class TMaskImage>
const TDisparityImage *
DisparityMapTo3DFilter<TDisparityImage,TOutputImage,TEpipolarGridImage,TMaskImage>
::GetHorizontalDisparityMapInput() const
{
  if(this->GetNumberOfInputs()<1)
    {
    return nullptr;
    }
  return static_cast<const TDisparityImage *>(this->itk::ProcessObject::GetInput(0));
}

template <class TDisparityImage, class TOutputImage,
class TEpipolarGridImage, class TMaskImage>
const TDisparityImage *
DisparityMapTo3DFilter<TDisparityImage,TOutputImage,TEpipolarGridImage,TMaskImage>
::GetVerticalDisparityMapInput() const
{
  if(this->GetNumberOfInputs()<2)
    {
    return nullptr;
    }
  return static_cast<const TDisparityImage *>(this->itk::ProcessObject::GetInput(1));
}

template <class TDisparityImage, class TOutputImage,
class TEpipolarGridImage, class TMaskImage>
const TEpipolarGridImage *
DisparityMapTo3DFilter<TDisparityImage,TOutputImage,TEpipolarGridImage,TMaskImage>
::GetLeftEpipolarGridInput() const
{
  if(this->GetNumberOfInputs()<3)
    {
    return nullptr;
    }
  return static_cast<const TEpipolarGridImage *>(this->itk::ProcessObject::GetInput(2));
}

template <class TDisparityImage, class TOutputImage,
class TEpipolarGridImage, class TMaskImage>
const TEpipolarGridImage *
DisparityMapTo3DFilter<TDisparityImage,TOutputImage,TEpipolarGridImage,TMaskImage>
::GetRightEpipolarGridInput() const
{
  if(this->GetNumberOfInputs()<4)
    {
    return nullptr;
    }
  return static_cast<const TEpipolarGridImage *>(this->itk::ProcessObject::GetInput(3));
}

template <class TDisparityImage, class TOutputImage,
class TEpipolarGridImage, class TMaskImage>
const TMaskImage *
DisparityMapTo3DFilter<TDisparityImage,TOutputImage,TEpipolarGridImage,TMaskImage>
::GetDisparityMaskInput() const
{
  if(this->GetNumberOfInputs()<5)
    {
    return nullptr;
    }
  return static_cast<const TMaskImage *>(this->itk::ProcessObject::GetInput(4));
}

template <class TDisparityImage, class TOutputImage,
class TEpipolarGridImage, class TMaskImage>
void
DisparityMapTo3DFilter<TDisparityImage,TOutputImage,TEpipolarGridImage,TMaskImage>
::GenerateOutputInformation()
{
  const TDisparityImage * horizDisp = this->GetHorizontalDisparityMapInput();
  TOutputImage * outputPtr = this->GetOutput();

  outputPtr->SetLargestPossibleRegion(horizDisp->GetLargestPossibleRegion());
  outputPtr->SetNumberOfComponentsPerPixel(3);

  // copy also origin and spacing
  outputPtr->SetOrigin(horizDisp->GetOrigin());
  outputPtr->SetSignedSpacing(horizDisp->GetSignedSpacing());
}

template <class TDisparityImage, class TOutputImage,
class TEpipolarGridImage, class TMaskImage>
void
DisparityMapTo3DFilter<TDisparityImage,TOutputImage,TEpipolarGridImage,TMaskImage>
::GenerateInputRequestedRegion()
{
  // For the epi grid : generate full buffer here !
  TEpipolarGridImage * leftGrid = const_cast<TEpipolarGridImage*>(this->GetLeftEpipolarGridInput());
  TEpipolarGridImage * rightGrid = const_cast<TEpipolarGridImage*>(this->GetRightEpipolarGridInput());

  leftGrid->SetRequestedRegionToLargestPossibleRegion();
  rightGrid->SetRequestedRegionToLargestPossibleRegion();

  TOutputImage * outputDEM = this->GetOutput();

  TDisparityImage * horizDisp = const_cast<TDisparityImage*>(this->GetHorizontalDisparityMapInput());
  TDisparityImage * vertiDisp = const_cast<TDisparityImage*>(this->GetVerticalDisparityMapInput());
  TMaskImage * maskDisp = const_cast<TMaskImage*>(this->GetDisparityMaskInput());

  // We impose that both disparity map inputs have the same size
  if(vertiDisp &&
     horizDisp->GetLargestPossibleRegion() != vertiDisp->GetLargestPossibleRegion())
    {
    itkExceptionMacro(<<"Horizontal and vertical disparity maps do not have the same size ! Horizontal largest region: "
      <<horizDisp->GetLargestPossibleRegion()<<", vertical largest region: "<<vertiDisp->GetLargestPossibleRegion());
    }


  if (maskDisp && horizDisp->GetLargestPossibleRegion() != maskDisp->GetLargestPossibleRegion())
    {
    itkExceptionMacro(<<"Disparity map and mask do not have the same size ! Map region : "
      <<horizDisp->GetLargestPossibleRegion()<<", mask region : "<<maskDisp->GetLargestPossibleRegion());
    }

  horizDisp->SetRequestedRegion( outputDEM->GetRequestedRegion() );

  if (vertiDisp)
    {
    vertiDisp->SetRequestedRegion( outputDEM->GetRequestedRegion() );
    }

  if (maskDisp)
    {
    maskDisp->SetRequestedRegion( outputDEM->GetRequestedRegion() );
    }

  // Check that the keywordlists are not empty
  if (m_LeftKeywordList.GetSize() == 0 || m_RightKeywordList.GetSize() == 0)
    {
    itkExceptionMacro(<<"At least one of the image keywordlist is empty : can't instantiate corresponding projection");
    }
}

template <class TDisparityImage, class TOutputImage,
class TEpipolarGridImage, class TMaskImage>
void
DisparityMapTo3DFilter<TDisparityImage,TOutputImage,TEpipolarGridImage,TMaskImage>
::BeforeThreadedGenerateData()
{
  // Instantiate transforms
  m_LeftToGroundTransform = RSTransformType::New();
  m_RightToGroundTransform = RSTransformType::New();

  m_LeftToGroundTransform->SetInputKeywordList(m_LeftKeywordList);

  m_RightToGroundTransform->SetInputKeywordList(m_RightKeywordList);
  m_LeftToGroundTransform->InstantiateTransform();
  m_RightToGroundTransform->InstantiateTransform();
}

template <class TDisparityImage, class TOutputImage,
class TEpipolarGridImage, class TMaskImage>
void
DisparityMapTo3DFilter<TDisparityImage,TOutputImage,TEpipolarGridImage,TMaskImage>
::ThreadedGenerateData(const RegionType & itkNotUsed(outputRegionForThread), itk::ThreadIdType itkNotUsed(threadId))
{
  const TDisparityImage * horizDisp = this->GetHorizontalDisparityMapInput();
  const TDisparityImage * vertiDisp = this->GetVerticalDisparityMapInput();

  const TMaskImage * disparityMask = this->GetDisparityMaskInput();

  TOutputImage * outputDEM = this->GetOutput();

  // Get epipolar grids
  const TEpipolarGridImage * leftGrid = this->GetLeftEpipolarGridInput();
  const TEpipolarGridImage * rightGrid = this->GetRightEpipolarGridInput();

  typename TOutputImage::RegionType outputRequestedRegion = outputDEM->GetRequestedRegion();

  // Define iterators
  itk::ImageRegionIterator<OutputImageType> demIt(outputDEM,outputRequestedRegion);
  itk::ImageRegionConstIteratorWithIndex<DisparityMapType> horizIt(horizDisp,outputRequestedRegion);

  demIt.GoToBegin();
  horizIt.GoToBegin();

  bool useVerti = false;
  itk::ImageRegionConstIteratorWithIndex<DisparityMapType> vertiIt;
  if (vertiDisp)
  {
    useVerti = true;
    vertiIt = itk::ImageRegionConstIteratorWithIndex<DisparityMapType>(vertiDisp,outputRequestedRegion);
    vertiIt.GoToBegin();
  }

  bool useMask = false;
  itk::ImageRegionConstIterator<MaskImageType> maskIt;
  if (disparityMask)
    {
    useMask = true;
    maskIt = itk::ImageRegionConstIterator<MaskImageType>(disparityMask,outputRequestedRegion);
    maskIt.GoToBegin();
    }

  // Avoid allocation for each loop
  typename OutputImageType::PixelType pixel3D(3);

  // Fill masked pixels if mask is set
  while (!demIt.IsAtEnd() && !horizIt.IsAtEnd())
    {
    if (useMask && !(maskIt.Get() > 0))
      {
      // TODO : what to do when masked ? put a no-data value ?
      typename OutputImageType::PixelType pixel3D(3);
      pixel3D.Fill(0);
      demIt.Set(pixel3D);
      
      ++demIt;
      ++horizIt;
      if (useVerti) ++vertiIt;
      ++maskIt;
      continue;
      }
    // compute left LOS
    typename TDisparityImage::PointType leftEpiPoint;
    horizDisp->TransformIndexToPhysicalPoint(horizIt.GetIndex(),leftEpiPoint);
    
    // Invert stereo-rectification grid
    auto leftSensorPoint = cars_details::InvertStereoRectificationGrid(leftEpiPoint, leftGrid);
    
    // Generate line of sight
    auto leftLos = cars_details::ComputeLineOfSight(leftSensorPoint, m_LeftToGroundTransform.GetPointer(), m_LeftMinimumElevation, m_LeftMaximumElevation);

    // compute right los

    // Apply disparity to index, and optionally vertical shift
    itk::ContinuousIndex<double,2> rightIndexEstimate;
    rightIndexEstimate[0] = static_cast<double>((horizIt.GetIndex())[0]) + static_cast<double>(horizIt.Get());
    
    double verticalShift = 0;
    if (useVerti) verticalShift = static_cast<double>(vertiIt.Get());
    rightIndexEstimate[1] = static_cast<double>((horizIt.GetIndex())[1]) + verticalShift;
    
    typename TDisparityImage::PointType rightEpiPoint;
    horizDisp->TransformContinuousIndexToPhysicalPoint(rightIndexEstimate,rightEpiPoint);
    
    // Invert stereo-rectification grid
    auto rightSensorPoint = cars_details::InvertStereoRectificationGrid(rightEpiPoint, rightGrid);
    
    // Generate line of sight
    auto rightLos = cars_details::ComputeLineOfSight(rightSensorPoint, m_RightToGroundTransform.GetPointer(), m_RightMinimumElevation, m_RightMaximumElevation);
    
    // Compute LOS intersection
    auto outPoint = cars_details::Intersect(leftLos, rightLos);
    
    // Transform ECEF to WGS84
    auto outPointWGS84 = cars_details::ToWGS84(outPoint);
    
    // record 3D point
    pixel3D[0] = outPointWGS84[0];
    pixel3D[1] = outPointWGS84[1];
    pixel3D[2] = outPointWGS84[2];
    demIt.Set(pixel3D);
    
    ++demIt;
    ++horizIt;
    
    if (useVerti) ++vertiIt;
    if (useMask) ++maskIt;   
    }
}

}

#endif
