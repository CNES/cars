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

#ifndef otbOptiDisparityMapTo3DFilter_h
#define otbOptiDisparityMapTo3DFilter_h

#include "itkImageToImageFilter.h"
#include "otbGenericRSTransform.h"
#include "otbVectorImage.h"
#include "otbImage.h"

#include <ossim/base/ossimEcefPoint.h>
#include <ossim/base/ossimGpt.h>
#include <vnl/vnl_inverse.h>

namespace otb
{ 
  namespace cars_details
  {
  template <typename TPoint, typename TGrid>
  TPoint
  InvertStereoRectificationGrid(const TPoint & epipolarPoint,
                                const TGrid * stereorectificationGrid)
  {

    itk::ContinuousIndex<double,2> gridIndexConti;

    // Get epipolar point coordinates in grid
    stereorectificationGrid->TransformPhysicalPointToContinuousIndex(epipolarPoint,gridIndexConti);
    
    // Get index of upper left pixel in grid
    typename TGrid::IndexType ulIndex, urIndex, lrIndex, llIndex;
    ulIndex[0] = static_cast<int>(std::floor(gridIndexConti[0]));
    ulIndex[1] = static_cast<int>(std::floor(gridIndexConti[1]));

    // Make sure index is in grid region (not sure this test is really
    // necessary ?)
    auto gridRegion = stereorectificationGrid->GetLargestPossibleRegion();
    if (ulIndex[0] < gridRegion.GetIndex(0)) ulIndex[0] = gridRegion.GetIndex(0);
    if (ulIndex[1] < gridRegion.GetIndex(1)) ulIndex[1] = gridRegion.GetIndex(1);
    if (ulIndex[0] > (gridRegion.GetIndex(0) + static_cast<int>(gridRegion.GetSize(0)) - 2))
      {
      ulIndex[0] = gridRegion.GetIndex(0) + gridRegion.GetSize(0) - 2;
      }
    if (ulIndex[1] > (gridRegion.GetIndex(1) + static_cast<int>(gridRegion.GetSize(1)) - 2))
      {
      ulIndex[1] = gridRegion.GetIndex(1) + gridRegion.GetSize(1) - 2;
      }

    // Build index of upper right, lower right and lower left pixels
    // in grid
    urIndex[0] = ulIndex[0] + 1;
    urIndex[1] = ulIndex[1];
    lrIndex[0] = ulIndex[0] + 1;
    lrIndex[1] = ulIndex[1] + 1;
    llIndex[0] = ulIndex[0];
    llIndex[1] = ulIndex[1] + 1;

    // Derive sub-pixel position
    itk::ContinuousIndex<double,2>  subPixIndex;
    subPixIndex[0] = gridIndexConti[0] - static_cast<double>(ulIndex[0]);
    subPixIndex[1] = gridIndexConti[1] - static_cast<double>(ulIndex[1]);

    // Transform to grid physical space
    typename TGrid::PointType ulPoint, urPoint, lrPoint, llPoint;
    stereorectificationGrid->TransformIndexToPhysicalPoint(ulIndex, ulPoint);
    stereorectificationGrid->TransformIndexToPhysicalPoint(urIndex, urPoint);
    stereorectificationGrid->TransformIndexToPhysicalPoint(lrIndex, lrPoint);
    stereorectificationGrid->TransformIndexToPhysicalPoint(llIndex, llPoint);

    // Apply deformation field to all 4 points to get coordinates in
    // raw image
    TPoint ulPixel, urPixel, lrPixel, llPixel, cPixel;
    ulPixel[0] = (stereorectificationGrid->GetPixel(ulIndex))[0] + ulPoint[0];
    ulPixel[1] = (stereorectificationGrid->GetPixel(ulIndex))[1] + ulPoint[1];
    urPixel[0] = (stereorectificationGrid->GetPixel(urIndex))[0] + urPoint[0];
    urPixel[1] = (stereorectificationGrid->GetPixel(urIndex))[1] + urPoint[1];
    lrPixel[0] = (stereorectificationGrid->GetPixel(lrIndex))[0] + lrPoint[0];
    lrPixel[1] = (stereorectificationGrid->GetPixel(lrIndex))[1] + lrPoint[1];
    llPixel[0] = (stereorectificationGrid->GetPixel(llIndex))[0] + llPoint[0];
    llPixel[1] = (stereorectificationGrid->GetPixel(llIndex))[1] + llPoint[1];

    // Derive coordinate of center pixel by interpolation
    cPixel[0] = (ulPixel[0] * (1.0 - subPixIndex[0]) + urPixel[0] * subPixIndex[0]) * (1.0 - subPixIndex[1]) + (llPixel[0] * (1.0 - subPixIndex[0]) + lrPixel[0] * subPixIndex[0]) * subPixIndex[1];
    cPixel[1] = (ulPixel[1] * (1.0 - subPixIndex[0]) + urPixel[1] * subPixIndex[0]) * (1.0 - subPixIndex[1]) + (llPixel[1] * (1.0 - subPixIndex[0]) + lrPixel[1] * subPixIndex[0]) * subPixIndex[1];

    return cPixel;
  }

  template<typename TPoint> TPoint ToEcef(const TPoint & in)
  {
    ossimGpt gpt(in[1], in[0], in[2]);
    ossimEcefPoint ecef(gpt);

    TPoint ret;
    ret[0] = ecef.x();
    ret[1] = ecef.y();
    ret[2] = ecef.z();
      
    return ret;
  }

  template <typename TPoint>  TPoint ToWGS84(const TPoint & in)
  {
    ossimEcefPoint ecef(in[0], in[1], in[2]);
    
    ossimGpt gpt(ecef);
    
    TPoint ret;
    ret[0] = gpt.lond();
    ret[1] = gpt.latd();
    ret[2] = gpt.height();
    
    return ret;
  }


  template <typename TPoint, typename TTransform, typename TOutputPoint = typename TTransform::OutputPointType>
  std::pair<TOutputPoint, TOutputPoint>
  ComputeLineOfSight(const TPoint & sensorPoint,
                     const TTransform * sensorToGroundTransform,
                     double elevationMin = 0,
                     double elevationMax = 300)
  {
    // Build sensor point at hmin
    typename TTransform::InputPointType sensorPoint3D;
    sensorPoint3D[0] = sensorPoint[0];
    sensorPoint3D[1] = sensorPoint[1];
    sensorPoint3D[2] = elevationMin;
    
    // LOS origin point in ECEF coordinates
    auto leftGroundHmin = ToEcef(sensorToGroundTransform->TransformPoint(sensorPoint3D));

    // Switch to hmax
    sensorPoint3D[2] = elevationMax;

    // LOS destination in ECEF coordinates
    auto leftGroundHmax = ToEcef(sensorToGroundTransform->TransformPoint(sensorPoint3D));

    // Return pair
    return std::make_pair(leftGroundHmax, leftGroundHmin);
  }

  template <typename TPoint, typename TPrecision = double> TPoint Intersect(const std::pair<TPoint, TPoint> & los1,
                                                                            const std::pair<TPoint, TPoint> & los2)
  {
    // Variables used for optimization
    vnl_vector<TPrecision>  midPoint3D;
    vnl_matrix<TPrecision> invCumul(3,3);
    invCumul.fill(0);
    vnl_matrix<TPrecision> identity(3,3);
    identity.fill(0);
    identity.fill_diagonal(1.);
    vnl_vector<TPrecision> secCumul(3);
    secCumul.fill(0);
    vnl_matrix<TPrecision> idMinusViViT(3,3);
    vnl_matrix<TPrecision> vi(3,1);
    vnl_vector<TPrecision> si(3);
    TPrecision norm_inv;

    // Points A0 and B0
    auto A0 = los1.first;
    auto B0 = los1.second;
    vi(0,0) = B0[0] - A0[0];
    vi(1,0) = B0[1] - A0[1];
    vi(2,0) = B0[2] - A0[2];

    norm_inv = 1. / std::sqrt(vi(0,0)*vi(0,0)+vi(1,0)*vi(1,0)+vi(2,0)*vi(2,0));

    vi(0,0) *= norm_inv;
    vi(1,0) *= norm_inv;
    vi(2,0) *= norm_inv;

    si(0) = A0[0];
    si(1) = A0[1];
    si(2) = A0[2];

    idMinusViViT = identity - (vi * vi.transpose());

    invCumul+=idMinusViViT;
    secCumul+=(idMinusViViT * si);

    // Points A1 and B1
    auto A1 = los2.first;
    auto B1 = los2.second;
    vi(0,0) = B1[0] - A1[0];
    vi(1,0) = B1[1] - A1[1];
    vi(2,0) = B1[2] - A1[2];

    norm_inv = 1. / std::sqrt(vi(0,0)*vi(0,0)+vi(1,0)*vi(1,0)+vi(2,0)*vi(2,0));

    vi(0,0) *= norm_inv;
    vi(1,0) *= norm_inv;
    vi(2,0) *= norm_inv;

    si(0) = A1[0];
    si(1) = A1[1];
    si(2) = A1[2];

    idMinusViViT = identity - (vi * vi.transpose());

    invCumul+=idMinusViViT;
    secCumul+=(idMinusViViT * si);

    // Cmpute midPoint3D
    midPoint3D = vnl_inverse(invCumul) * secCumul;

    TPoint outPoint;
    outPoint[0] = midPoint3D[0];
    outPoint[1] = midPoint3D[1];
    outPoint[2] = midPoint3D[2];
    
    return outPoint;
  }

} // end namespace cars_details
 

/** \class DisparityMapTo3DFilter
 *  \brief Project an input disparity map into a 3D points
 *
 *  This filter uses an input disparity map (horizontal and vertical) to project 3D points.
 *  The output image contains the 3D points coordinates for each location of input disparity.
 *  The 3D coordinates (sorted by band) are : longitude , latitude (in degree, wrt WGS84) and altitude (in meters)
 *
 *  \sa FineRegistrationImageFilter
 *  \sa StereorectificationDisplacementFieldSource
 *  \sa SubPixelDisparityImageFilter
 *  \sa PixelWiseBlockMatchingImageFilter
 *
 *  \ingroup Streamed
 *  \ingroup Threaded
 *
 *
 * \ingroup OTBDisparityMap
 */
template <class TDisparityImage, class TOutputImage =  otb::VectorImage<float,2>,
          class TEpipolarGridImage = otb::VectorImage<float,2> , class TMaskImage = otb::Image<unsigned char> >
class ITK_EXPORT DisparityMapTo3DFilter :
    public itk::ImageToImageFilter<TDisparityImage,TOutputImage>
{
public:
  /** Standard class typedef */
  typedef DisparityMapTo3DFilter                            Self;
  typedef itk::ImageToImageFilter<TDisparityImage,
                                  TOutputImage>             Superclass;
  typedef itk::SmartPointer<Self>                           Pointer;
  typedef itk::SmartPointer<const Self>                     ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(DisparityMapTo3DFilter, ImageToImageFilter);

  /** Useful typedefs */
  typedef TDisparityImage         DisparityMapType;
  typedef TOutputImage            OutputImageType;
  typedef TEpipolarGridImage      GridImageType;
  typedef TMaskImage              MaskImageType;

  typedef typename OutputImageType::RegionType         RegionType;
  typedef typename OutputImageType::PixelType          DEMPixelType;

  // 3D RS transform
  // TODO: Allow tuning precision (i.e. double or float)
  typedef otb::GenericRSTransform<double,3,3>       RSTransformType;

  // 3D points
  typedef typename RSTransformType::InputPointType  TDPointType;

  typedef otb::ImageKeywordlist                     ImageKeywordListType;

  /** Set horizontal disparity map input */
  void SetHorizontalDisparityMapInput( const TDisparityImage * hmap);

  /** Set vertical disparity map input */
  void SetVerticalDisparityMapInput( const TDisparityImage * vmap);

  /** Set left epipolar grid (deformation grid from sensor image to epipolar space, regular in epipolar space)*/
  void SetLeftEpipolarGridInput( const TEpipolarGridImage * grid);

  /** Set right epipolar grid (deformation grid from sensor image to epipolar space, regular in epipolar space)*/
  void SetRightEpipolarGridInput( const TEpipolarGridImage * grid);

  /** Set mask associated to disparity maps (optional, pixels with a null mask value are ignored) */
  void SetDisparityMaskInput( const TMaskImage * mask);

  /** Get the inputs */
  const TDisparityImage * GetHorizontalDisparityMapInput() const;
  const TDisparityImage * GetVerticalDisparityMapInput() const;
  const TEpipolarGridImage * GetLeftEpipolarGridInput() const;
  const TEpipolarGridImage * GetRightEpipolarGridInput() const;
  const TMaskImage  * GetDisparityMaskInput() const;

  /** Set left keywordlist */
  void SetLeftKeywordList(const ImageKeywordListType kwl)
    {
    this->m_LeftKeywordList = kwl;
    this->Modified();
    }

  /** Get left keywordlist */
  const ImageKeywordListType & GetLeftKeywordList() const
    {
    return this->m_LeftKeywordList;
    }

   /** Set right keywordlist */
  void SetRightKeywordList(const ImageKeywordListType kwl)
    {
    this->m_RightKeywordList = kwl;
    this->Modified();
    }

  /** Get right keywordlist */
  const ImageKeywordListType & GetRightKeywordList() const
    {
    return this->m_RightKeywordList;
    }

  itkSetMacro(LeftMinimumElevation,double);
  itkSetMacro(LeftMaximumElevation,double);
  itkGetConstReferenceMacro(LeftMinimumElevation,double);
  itkGetConstReferenceMacro(LeftMaximumElevation,double);

  itkSetMacro(RightMinimumElevation,double);
  itkSetMacro(RightMaximumElevation,double);
  itkGetConstReferenceMacro(RightMinimumElevation,double);
  itkGetConstReferenceMacro(RightMaximumElevation,double);


protected:
  /** Constructor */
  DisparityMapTo3DFilter();

  /** Destructor */
  ~DisparityMapTo3DFilter() override;

  /** Generate output information */
  void GenerateOutputInformation() override;

  /** Generate input requested region */
  void GenerateInputRequestedRegion() override;

  /** Before threaded generate data */
  void BeforeThreadedGenerateData() override;

  /** Threaded generate data */
  void ThreadedGenerateData(const RegionType & outputRegionForThread, itk::ThreadIdType threadId) override;

  /** Override VerifyInputInformation() since this filter's inputs do
    * not need to occupy the same physical space.
    *
    * \sa ProcessObject::VerifyInputInformation
    */
  void VerifyInputInformation() override {}


private:
  DisparityMapTo3DFilter(const Self&) = delete;
  void operator=(const Self&) = delete;

  /** Keywordlist of left sensor image */
  ImageKeywordListType m_LeftKeywordList;

  /** Keywordlist of right sensor image */
  ImageKeywordListType m_RightKeywordList;

  /** Left sensor image transform */
  RSTransformType::Pointer m_LeftToGroundTransform;

  /** Right sensor image transform */
  RSTransformType::Pointer m_RightToGroundTransform;

  /** Elevation range for LOS generation */
  double m_LeftMinimumElevation;
  double m_LeftMaximumElevation;
  double m_RightMinimumElevation;
  double m_RightMaximumElevation;


};
} // end namespace otb

#ifndef OTB_MANUAL_INSTANTIATION
#include "otbOptiDisparityMapTo3DFilter.hxx"
#endif

#endif
