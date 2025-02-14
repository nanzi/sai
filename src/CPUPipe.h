/*
    This file is part of SAI, which is a fork of Leela Zero.
    Copyright (C) 2017-2018 Junhee Yoo and contributors
    Copyright (C) 2019 SAI Team

    SAI is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    SAI is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with SAI.  If not, see <http://www.gnu.org/licenses/>.

    Additional permission under GNU GPL version 3 section 7

    If you modify this Program, or any covered work, by linking or
    combining it with NVIDIA Corporation's libraries from the
    NVIDIA CUDA Toolkit and/or the NVIDIA CUDA Deep Neural
    Network library and/or the NVIDIA TensorRT inference library
    (or a modified version of those libraries), containing parts covered
    by the terms of the respective license agreement, the licensors of
    this Program grant you additional permission to convey the resulting
    work.
*/

#ifndef CPUPIPE_H_INCLUDED
#define CPUPIPE_H_INCLUDED
#include "config.h"

#include <vector>
#include <cassert>

#include "ForwardPipe.h"

class CPUPipe : public ForwardPipe {
public:
    virtual void initialize(const int channels);
    virtual void forward(const std::vector<float>& input,
                         std::vector<float>& output_pol,
                         std::vector<float>& output_val);

    virtual void push_weights(unsigned int filter_size,
                              unsigned int channels,
                              unsigned int outputs,
                              std::shared_ptr<const ForwardPipeWeights> weights);
private:
    void winograd_transform_in(const std::vector<float>& in,
                               std::vector<float>& V,
                               const int C);

    void winograd_sgemm(const std::vector<float>& U,
                        const std::vector<float>& V,
                        std::vector<float>& M,
                        const int C, const int K);

    void winograd_transform_out(const std::vector<float>& M,
                                std::vector<float>& Y,
                                const int K);

    void winograd_convolve3(const int outputs,
                            const std::vector<float>& input,
                            const std::vector<float>& U,
                            std::vector<float>& V,
                            std::vector<float>& M,
                            std::vector<float>& output);


    int m_input_channels;

    // Input + residual block tower
    std::shared_ptr<const ForwardPipeWeights> m_weights;
};
#endif
