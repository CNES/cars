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
#ifndef otbVlfeatSiftImageFilter_hxx
#define otbVlfeatSiftImageFilter_hxx

#include "otbVlfeatSiftImageFilter.h"

#include <vl/sift.h>
#include <cmath>

#include "itkContinuousIndex.h"
#include "itkImageRegionConstIterator.h"

namespace otb
{
/**
 * Constructor
 */
template <class TInputImage, class TOutputPointSet>
VlfeatSiftImageFilter<TInputImage, TOutputPointSet>
::VlfeatSiftImageFilter() : m_DoGThreshold(0.04), m_NumberOfOctaves(3), m_NumberOfScalesPerOctave(1),
    m_EdgeThreshold(10.0), m_Magnification(3.0)
{}

/**
 * Converts DoG threshold wrt scale-space.
 */
template <class TInputImage, class TOutputPointSet>
float
VlfeatSiftImageFilter<TInputImage, TOutputPointSet>
::ConvertDogThreshold() const
{
    float k_nspo = exp(M_LN2 / m_NumberOfScalesPerOctave);
    float k_3 = exp(M_LN2 / 3.0f);
    return (k_nspo - 1.0f) / (k_3 - 1.0f) * m_DoGThreshold;
}

template <class TInputImage, class TOutputPointSet>
void
VlfeatSiftImageFilter<TInputImage, TOutputPointSet>
::GenerateData()
{

  // Get the input image pointer
  const InputImageType *    inputPtr       = this->GetInput();
  OutputPointSetPointerType outputPointSet = this->GetOutput();

  typename InputImageType::SizeType size = inputPtr->GetLargestPossibleRegion().GetSize();


  // TODO: mention vlfeat copyright on this file, since some code as
  // been imported from vlfeat/src/sift.c
  
  // Initialize the vlfeat class that does the computation
  int omin = -1;
  VlSiftFilt * filt = vl_sift_new (size[0], size[1], m_NumberOfOctaves, m_NumberOfScalesPerOctave, omin);

  if (m_EdgeThreshold >= 0) vl_sift_set_edge_thresh (filt, m_EdgeThreshold) ;
  if (m_DoGThreshold >= 0) vl_sift_set_peak_thresh (filt, ConvertDogThreshold()) ;
  if (m_Magnification >= 0) vl_sift_set_magnif(filt, m_Magnification) ;
  
  int err = 0;

  // This id identifies keypoints
  unsigned int id = 0;
  
  for(unsigned int octave = 0; octave < m_NumberOfOctaves; ++octave)
    {
    // First call is different
    if(octave == 0)
      {
      err = vl_sift_process_first_octave(filt,inputPtr->GetBufferPointer());
      }
    else
      {
      err = vl_sift_process_next_octave(filt);
      }
    // TODO: handle error code err here 

    // Detect keypoints
    vl_sift_detect(filt);
    
    VlSiftKeypoint const * keys = vl_sift_get_keypoints(filt);
    int nkeys = vl_sift_get_nkeypoints(filt);

    // Loop on keypoints
    for (unsigned int i = 0; i < nkeys ; ++i)
      {
      double angles [4];
      int nangles;
      VlSiftKeypoint const *k;
      k = keys + i;
      
      // Calculate orientation
      nangles = vl_sift_calc_keypoint_orientations(filt, angles, k);
      
      // Loop on orientation
      for (unsigned int q = 0 ; q < (unsigned) nangles ; ++q)
        {
        // Compute descriptor
        vl_sift_pix descr [128];
        vl_sift_calc_keypoint_descriptor(filt, descr, k, angles [q]);
        
        // Get the key location
        itk::ContinuousIndex<float, 2> keyContIndex;
        keyContIndex[0] = k->x;
        keyContIndex[1] = k->y;
        
        // Transform to image space
        OutputPointType point;
        inputPtr->TransformContinuousIndexToPhysicalPoint(keyContIndex, point);
        
        // Get the key descriptor
        OutputPixelType data;
        data.SetSize(128);
        for (int descIt = 0; descIt < 128; ++descIt)
          {
          data[descIt] = descr[descIt];
          }
        
        outputPointSet->SetPoint(id, point);
        outputPointSet->SetPointData(id, data);
        ++id;
        }
      }
    }

  if (filt)
    {
    vl_sift_delete (filt) ;
    filt = 0 ;
    }
}
/*
 * PrintSelf Method
 */
template <class TInputImage, class TOutputPointSet>
void
VlfeatSiftImageFilter<TInputImage, TOutputPointSet>
::PrintSelf(std::ostream& os, itk::Indent indent) const
{
  Superclass::PrintSelf(os, indent);
}

} // End namespace otb

#endif
