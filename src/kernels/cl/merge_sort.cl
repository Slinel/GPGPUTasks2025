#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* from,
    __global       uint* to,
                   uint  k,
                   uint  n)
{
    uint gid = get_global_id(0);
    if(gid<n) {
        uint self_value = from[gid];
        uint block_idx = gid/k;
        uint dir_to_neighbor = 1-2*((block_idx)%2);
        uint l = min(n, (block_idx  +dir_to_neighbor)*k);
        uint r = min(n, (block_idx+1+dir_to_neighbor)*k);
        while (l<r) {
            uint m = l + (r-l)/2;
            // придумал как сделать бранчлесс
            if (from[m]<self_value + (block_idx&1)) {
                l = m+1;
            } else {
                r = m;
            }
        }
        to[gid + l-min(n, (block_idx  +dir_to_neighbor)*k)-k*(block_idx%2)] = self_value;
    }
}
