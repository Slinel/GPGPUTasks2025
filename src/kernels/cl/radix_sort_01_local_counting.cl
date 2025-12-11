#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* from,
    __global       uint* to,
    unsigned int letter_idx,
    unsigned int n,
    short int letter_looking_for)
{
    unsigned int gid = get_global_id(0);
    if(gid<n) {
        unsigned int mask = ((1<<letter_len_in_bits)-1)<<letter_idx;
        unsigned int letter_found = ((from[gid])&mask)>>letter_idx;
        to[gid]=(letter_looking_for==letter_found);
    }
    return;
}
