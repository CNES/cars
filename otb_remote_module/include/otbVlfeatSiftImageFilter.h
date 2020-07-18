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

#ifndef otbVlfeatSiftImageFilter_h
#define otbVlfeatSiftImageFilter_h

#include "otbImageToPointSetFilter.h"
#include "otbImage.h"
//#include "itkMacro.h"

#ifndef M_LN2
    #define M_LN2 0.693147180559945309417
#endif

namespace otb
{

/** \class VlfeatSiftImageFilter
 *  \brief This class extracts key points from an input image through a pyramidal decomposition
 * \sa ImageToSIFTKeyPointSetFilter
 *
 * \ingroup OTBDescriptors
 */
template <class TInputImage, class TOutputPointSet>
class ITK_EXPORT VlfeatSiftImageFilter
  : public ImageToPointSetFilter<TInputImage, TOutputPointSet>
{
public:
  /** Standard typedefs */
  typedef VlfeatSiftImageFilter                                 Self;
  typedef ImageToPointSetFilter<TInputImage, TOutputPointSet> Superclass;
  typedef itk::SmartPointer<Self>                             Pointer;
  typedef itk::SmartPointer<const Self>                       ConstPointer;

  /** Creation through object factory macro */
  itkNewMacro(Self);

  /** Type macro */
  itkTypeMacro(VlfeatSiftImageFilter, ImageToPointSetFilter);

  /** Template parameters typedefs */

  typedef TInputImage                     InputImageType;
  typedef typename TInputImage::Pointer   InputImagePointerType;
  typedef typename TInputImage::PixelType PixelType;

  typedef TOutputPointSet                           OutputPointSetType;
  typedef typename TOutputPointSet::Pointer         OutputPointSetPointerType;
  typedef typename TOutputPointSet::PixelType       OutputPixelType;
  typedef typename TOutputPointSet::PointType       OutputPointType;
  typedef typename TOutputPointSet::PointIdentifier OutputPointIdentifierType;

  typedef otb::Image<float, 2>                             FloatImageType;
  typedef std::vector<std::pair<OutputPointType, double> > OrientationVectorType;

  itkSetMacro(NumberOfOctaves, int);
  itkSetMacro(NumberOfScalesPerOctave, int);
  itkSetMacro(DoGThreshold, float);
  itkSetMacro(EdgeThreshold, float);
  itkSetMacro(Magnification, float);

protected:
  /** Actually process the input */
  void GenerateData() override;

  /** Constructor */
  VlfeatSiftImageFilter();

  /** Destructor */
  ~VlfeatSiftImageFilter() override {}

  /** PrintSelf method */
  void PrintSelf(std::ostream& os, itk::Indent indent) const override;

  float ConvertDogThreshold() const;

private:
  float m_DoGThreshold;
  float m_EdgeThreshold;
  float m_Magnification;
  unsigned int m_NumberOfOctaves;
  unsigned int m_NumberOfScalesPerOctave;
  float m_MatchingThreshold;

};
} // End namespace otb
#ifndef OTB_MANUAL_INSTANTIATION
#include "otbVlfeatSiftImageFilter.hxx"
#endif

#endif
