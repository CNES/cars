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
#include "otbFunctorImageFilter.h"

namespace otb
{

namespace Wrapper
{

class BuildMask : public otb::Wrapper::Application
{
public:
  typedef BuildMask Self;
  typedef itk::SmartPointer<Self> Pointer; 

  itkNewMacro(Self);
  itkTypeMacro(BuildMask, otb::Wrapper::Application);

private:
  void DoInit()
  {
    SetName("BuildMask");
    SetDescription("Build a mask of pixels to ignore");

    SetDocLimitations("None");
    SetDocAuthors("OTB-Team");

    AddParameter(ParameterType_InputImage, "in", "Input Image");
    SetParameterDescription("in"," Input image");

    AddParameter(ParameterType_InputImage, "inmask", "Input Mask");
    SetParameterDescription("inmask"," Input mask");
    MandatoryOff("inmask");
      
    AddParameter(ParameterType_Float,"innodata","No data value for input image");
    SetParameterDescription("innodata","No data value for input image");
    SetDefaultParameterFloat("innodata",0);

    AddParameter(ParameterType_Float,"outnodata","No data value for input image");
    SetParameterDescription("outnodata","No data value for input image");
    SetDefaultParameterFloat("outnodata",255);

    AddParameter(ParameterType_Float,"outvalid","No data value for input image");
    SetParameterDescription("outvalid","No data value for input image");
    SetDefaultParameterFloat("outvalid",0);



    AddParameter(ParameterType_OutputImage, "out", "Mask of ignored pixels");
  }

  void DoUpdateParameters()
  {}
  
  void DoExecute()
  {  
    auto innodata = GetParameterFloat("innodata");
    auto outnodata = GetParameterFloat("outnodata");
    auto outvalid = GetParameterFloat("outvalid");

    auto lambdaWithMask = [innodata,outnodata,outvalid](const float & pixel_value, const short & mask_value){return pixel_value != innodata ? (mask_value != 0 ? mask_value : outvalid) : outnodata;};
    auto filterWithMask = NewFunctorFilter(lambdaWithMask);

    auto lambda = [innodata,outnodata,outvalid](const float & pixel_value){return pixel_value != innodata ? outvalid : outnodata;};
    auto filter = NewFunctorFilter(lambda);

    if(IsParameterEnabled("inmask"))
      {
      filterWithMask->SetInput<0>(GetParameterFloatImage("in"));
      filterWithMask->SetInput<1>(GetParameterInt16Image("inmask"));
      SetParameterOutputImage("out",filterWithMask->GetOutput());
      m_Filter = filterWithMask;
      }
    else
      {
      filter->SetInput<0>(GetParameterFloatImage("in"));
      SetParameterOutputImage("out",filter->GetOutput());
      m_Filter = filter;
      }
  }

  itk::LightObject::Pointer m_Filter;
  
  };

}
}

OTB_APPLICATION_EXPORT(otb::Wrapper::BuildMask)
