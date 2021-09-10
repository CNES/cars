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
#include "itkImageRegionIteratorWithIndex.h"
#include "otbWrapperElevationParametersHandler.h"
#include "otbDEMHandler.h"

namespace otb
{

namespace Wrapper
{

class DEMReader : public otb::Wrapper::Application
{
public:
  typedef DEMReader Self;
  typedef itk::SmartPointer<Self> Pointer; 

  itkNewMacro(Self);
  itkTypeMacro(DEMReader, otb::Wrapper::Application);

private:
  void DoInit()
  {
    SetName("DEMReader");
    SetDescription("Read a DEM extract using DEM Directory");

    SetDocLimitations("None");
    SetDocAuthors("OTB-Team");
    SetDocSeeAlso("");

    AddDocTag(Tags::Stereo);

    AddParameter(ParameterType_Float,"originx", "X origin of extract");
    SetParameterDescription("originx", "Grid x origin");

    AddParameter(ParameterType_Float,"originy", "Y origin of extract");
    SetParameterDescription("originy", "Grid y origin");

    AddParameter(ParameterType_Int,"sizex", "X size of extract");
    SetParameterDescription("sizex", "Grid x size");
    SetMinimumParameterIntValue("sizex",0);

    AddParameter(ParameterType_Int,"sizey", "Y size of extract");
    SetParameterDescription("sizey", "Grid y size");
    SetMinimumParameterIntValue("sizey",0);

    AddParameter(ParameterType_Float,"resolution", "Grid resolution");
    SetParameterDescription("resolution", "Grid resolution");
    SetMinimumParameterFloatValue("resolution",0);
    SetDefaultParameterFloat("resolution",0.000277777777778);

    ElevationParametersHandler::AddElevationParameters(this, "elev");

    AddParameter(ParameterType_OutputImage,"out","Out DEM extract");
    SetParameterDescription("out","Out DEM extract");
  }

  void DoUpdateParameters()
  {
    // Clear and reset the DEM Handler
    otb::DEMHandler::Instance()->ClearDEMs();
    otb::Wrapper::ElevationParametersHandler::SetupDEMHandlerFromElevationParameters(this, "elev");
  }

  void DoExecute()
  {
   // Clear and reset the DEM Handler
    otb::DEMHandler::Instance()->ClearDEMs();
    otb::Wrapper::ElevationParametersHandler::SetupDEMHandlerFromElevationParameters(this, "elev");

    auto output = FloatImageType::New();
    FloatImageType::SizeType size;
    size[0] = this->GetParameterInt("sizex");
    size[1] = this->GetParameterInt("sizey");
    FloatImageType::RegionType region;
    region.SetSize(size);

    output->SetRegions(region);
    output->Allocate();
    output->FillBuffer(-32768);

    FloatImageType::PointType origin;
    origin[0] = this->GetParameterFloat("originx");
    origin[1] = this->GetParameterFloat("originy");
    output->SetOrigin(origin);

    FloatImageType::SpacingType spacing;
    spacing[0] = this->GetParameterFloat("resolution");
    spacing[1] = -spacing[0];
    output->SetSpacing(spacing);

    itk::ImageRegionIteratorWithIndex<FloatImageType> it(output,region);

    it.GoToBegin();

    FloatImageType::PointType p;

    auto dem = otb::DEMHandler::Instance();

    while(!it.IsAtEnd())
      {
      output->TransformIndexToPhysicalPoint(it.GetIndex(),p);

      it.Set(dem->GetHeightAboveEllipsoid(p));
      
      ++it;
      }

    SetParameterOutputImage("out",output.GetPointer());
  }
};

}
}

OTB_APPLICATION_EXPORT(otb::Wrapper::DEMReader)
