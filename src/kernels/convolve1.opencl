/*
    This file is part of SAI, which is a fork of Leela Zero.
    Copyright (C) 2017-2019 Gian-Carlo Pascutto and contributors

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

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(
    __kernel
    __attribute__((work_group_size_hint(8, 8, 4)))
    void convolve1(
                   __global const net_t * restrict in,
                   __global net_t * restrict out,
                   __global const net_t * restrict weights,
                   __global const net_t * restrict means,
                   __global const net_t * restrict stdevs,
                   __local real * channel_buff,
                   __local real * filter_buff,
                   __local real * merge_buff,
                   __private const int4 sizes,
                   __private const int add_origin) {

        // cl::NDRange global(ceilMultiple(sites,         loc_sites),
        //                    ceilMultiple(outputs,       loc_outputs),
        //                    ceilMultiple(num_items_a_c, loc_num_items_a_c)),
        // cl::NDRange local(loc_sites, loc_outputs, loc_num_items_a_c);

        const int num_mults_item = 1 << LOG2_NUM_MULTS_PER_ITEM;

        const int s  = get_global_id(0);  // site    [0, sites - 1]
        const int o  = get_global_id(1);  // output  [0, outputs - 1]
        
        const int inter = s % NUM_INTERSECTIONS;
        const int batch = s / NUM_INTERSECTIONS;

        const int sites         = sizes.s0; // sites == (get_global_size(0) / NUM_INTERSECTIONS) * NUM_INTERSECTIONS
        const int outputs       = sizes.s1;
        // const int num_items_a_c = get_global_size(2);
        const int channels      = sizes.s2; // channels <= num_items_a_c << LOG2_NUM_MULTS_PER_ITEM

        const int ls  = get_local_id(0);
        const int lo  = get_local_id(1);
        const int lci = get_local_id(2);
        const int lc  = lci << LOG2_NUM_MULTS_PER_ITEM;

        const int loc_sites          = get_local_size(0);
        const int loc_outputs        = get_local_size(1);
        const int loc_num_items_a_c  = get_local_size(2);
        const int loc_channels       = loc_num_items_a_c << LOG2_NUM_MULTS_PER_ITEM;
        const int numTiles           = 1 + (channels-1) / loc_channels; // ceil(channels / loc_channels)

        real val = 0;
        for (int t=0 ; t<numTiles ; t++) {
            const int ci = t * loc_num_items_a_c + lci;
            const int c  = ci << LOG2_NUM_MULTS_PER_ITEM;

            // Copy the input locally
            if (s < sites) {
                if (num_mults_item <= loc_outputs) {
                    // Values of 'lo' are enough to copy the input data one location per item
                    const int indexChannel = lc + lo;
                    const int globalChannel = c + lo;
                    if (lo < num_mults_item && globalChannel < channels) {
                        channel_buff[indexChannel * loc_sites + ls]
                            = vload_net_t((batch * channels + globalChannel) * NUM_INTERSECTIONS + inter, in);
                    }
                } else {
                    // Values of 'lo' are not enough: each item will copy the same share of input values
                    const int ratio = num_mults_item / loc_outputs; // always a power of 2
#pragma unroll
                    for (int i = 0; i < ratio; ++i) {
                        const int indexChannel = lc + lo * ratio + i;
                        const int globalChannel = c + lo * ratio + i;
                        if (globalChannel < channels) {
                            channel_buff[indexChannel * loc_sites + ls]
                                = vload_net_t((batch * channels + globalChannel) * NUM_INTERSECTIONS + inter, in);
                        }
                    }
                }
            }

            // Copy the filter we are applying locally
            if (o < outputs) {
                const int indexChannel = lc + ls;
                const int globalChannel = c + ls;
                if (num_mults_item <= loc_sites) {
                    // Values of 'ls' are enough to copy the filter data one location per item
                    if (ls < num_mults_item && globalChannel < channels) {
                        filter_buff[lo * loc_channels + indexChannel]
                            = vload_net_t(o * channels + globalChannel, weights);
                    }
                } else {
                    // Values of 'ls' are not enough: each item will copy the same share of filter values
                    const int ratio = num_mults_item / loc_sites; // always a power of 2
#pragma unroll
                    for (int i = 0; i < ratio; ++i) {
                        const int indexChannel = lc + ls * ratio + i;
                        const int globalChannel = c + ls * ratio + i;
                        if (globalChannel < channels) {
                            filter_buff[lo * loc_channels + indexChannel]
                                = vload_net_t(o * channels + globalChannel, weights);
                        }
                    }
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            // run the num_mults_item operations of this item and put them in the local merge_buffer
            if (s < sites && o < outputs) {

#pragma unroll
                for (int i = 0; i < num_mults_item; ++i) {
                    const int indexChannel = lc + i;
                    const int globalChannel = c + i;
                    if (globalChannel < channels) {
                        val += filter_buff[lo * loc_channels + indexChannel] * channel_buff[indexChannel * loc_sites + ls];
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        const int merge_buff_offset   = (ls * loc_outputs + lo) * loc_num_items_a_c;
        merge_buff[merge_buff_offset + lci] = val;

        if (lci == 0 && o < outputs) {
            if (ls == 0) {
                filter_buff[2 * lo] = vload_net_t(o, means);
            }
            if (ls == 1) {
                filter_buff[2 * lo + 1] = vload_net_t(o, stdevs);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);


        // after all items have fillted their result in the merge_buffer
        // run the merge at work-group level and store the result in global memory
        if (lci == 0 && s < sites && o < outputs) {
            real sum = 0;
            
#pragma unroll
            for (int i = 0; i < loc_num_items_a_c; ++i) {
                sum += merge_buff[merge_buff_offset + i];
            }

            const real mean  = filter_buff[2*lo];
            const real stdev = filter_buff[2*lo+1];

            sum = (sum - mean) * stdev;       // Batch normalization

            const int out_offset = (batch * outputs + o) * NUM_INTERSECTIONS + inter;
            if (add_origin) {
                sum += vload_net_t(out_offset, out);
            }
            sum = sum > ZERO ? sum : ZERO;    // ReLU

            vstore_net_t(sum, out_offset, out);
        }
    }
    

#if 0
    __kernel
    __attribute__((work_group_size_hint(32, 4, 1)))
    void merge(
               __global const net_t * restrict means,
               __global const net_t * restrict stdevs,
               __global const net_t * restrict mrg,
               __global net_t * restrict out,
               __local real * bn_buff,
               __private const int4 sizes,
               __private const int add_origin) {
        
        // cl::NDRange(ceilMultiple(sites, loc_merge_sites), outputs, 1),
        // cl::NDRange(loc_merge_sites, loc_merge_outputs, 1));

        const int s  = get_global_id(0);
        const int o  = get_global_id(1);

        const int ls = get_local_id(0);
        const int lo = get_local_id(1);

        const int sites          = sizes.s0;
        const int outputs        = sizes.s1;
        const int merge_size_a_c = sizes.s3;

        if (s < sites) { // this is redundant!
            if (ls == 2 * lo) {
                bn_buff[ls] = vload_net_t(o, means);
            }
            if (ls == 2*lo + 1) {
                bn_buff[ls] = vload_net_t(o, stdevs);
            }
        }
            
        barrier(CLK_LOCAL_MEM_FENCE);

        if (s < sites && o < outputs) { // o < outputs is redundant!
            real sum = 0;
            
#pragma unroll
            for (int i = 0; i < merge_size_a_c; ++i) {
                sum += vload_net_t((o * sites + s) * merge_size_a_c + i, mrg);
            }

            const real mean  = bn_buff[2*lo];
            const real stdev = bn_buff[2*lo+1];

            sum = (sum - mean) * stdev;       // Batch normalization

            const int inter  = s % NUM_INTERSECTIONS;
            const int batch  = s / NUM_INTERSECTIONS;

            const int out_offset = (batch * outputs + o) * NUM_INTERSECTIONS + inter;
            if (add_origin) {
                sum += vload_net_t(out_offset, out);
            }
            sum = sum > ZERO ? sum : ZERO;    // ReLU

            vstore_net_t(sum, out_offset, out);
        }
    }
#endif
// End of the C++11 raw string literal
)"
