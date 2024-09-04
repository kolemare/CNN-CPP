/*
MIT License
Copyright (c) 2024 Marko Kostić

Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the "Software"), to deal 
in the Software without restriction, including without limitation the rights 
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

This project is the CNN-CPP Framework. Usage of this code is free, and 
uploading and using the code is also free, with a humble request to mention 
the origin of the implementation, the author Marko Kostić, and the repository 
link: https://github.com/kolemare/CNN-CPP.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
*/

#include "GradientClipping.hpp"

void GradientClipping::clipGradients(Eigen::Tensor<double, 4> &gradients,
                                     double clip_value)
{
    // Clip the values of the 4D tensor to lie within the range [-clip_value, clip_value]
    for (int i = 0; i < gradients.size(); ++i)
    {
        gradients.data()[i] = std::max(std::min(gradients.data()[i], clip_value), -clip_value);
    }
}

void GradientClipping::clipGradients(Eigen::Tensor<double, 2> &gradients,
                                     double clip_value)
{
    // Clip the values of the 2D tensor to lie within the range [-clip_value, clip_value]
    for (int i = 0; i < gradients.size(); ++i)
    {
        gradients.data()[i] = std::max(std::min(gradients.data()[i], clip_value), -clip_value);
    }
}
