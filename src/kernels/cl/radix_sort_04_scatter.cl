#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* from,
    __global       uint* to,
    unsigned int n,
    unsigned int letter_idx,
    __global const uint* pref0,
    __global const uint* pref1,
    __global const uint* pref2,
    __global const uint* pref3)
{
    unsigned int gid = get_global_id(0);
    if(gid<n) {
        unsigned int val = from[gid];
        unsigned int mask = ((1<<letter_len_in_bits)-1)<<letter_idx;
        unsigned int letter_value = (val&mask)>>letter_idx;
        unsigned int idx = 0;
        if(letter_value==0) {
            idx=pref0[gid]-1;
        } else if(letter_value==1) {
            idx=pref1[gid]-1;
            idx+=pref0[n-1];
        } else if(letter_value==2) {
            idx=pref2[gid]-1;
            idx+=pref0[n-1];
            idx+=pref1[n-1];
        } else if(letter_value==3) {
            idx=pref3[gid]-1;
            idx+=pref0[n-1];
            idx+=pref1[n-1];
            idx+=pref2[n-1];
        }
        //printf("gid: %d letter: %d idx: %d", gid, letter_value, idx);
        to[idx] = val;
    }
    return;
}