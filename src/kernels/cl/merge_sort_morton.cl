#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#include "../shared_structs/camera_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/aabb_gpu_shared.h"
#include "../shared_structs/morton_code_gpu_shared.h"

#include "camera_helpers.cl"
#include "geometry_helpers.cl"
#include "random_helpers.cl"


__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort_morton(
    __global const MortonCode* code_from,
    __global const uint*       indx_from,
    __global       MortonCode* code_to,
    __global       uint*       indx_to,
    uint                       k,
    uint                       n)
{
    uint gid = get_global_id(0);
    if(gid<n) {
        uint self_code_value = code_from[gid];
        uint self_indx_value = indx_from[gid];
        uint block_idx = gid/k;
        uint dir_to_neighbor = 1-2*((block_idx)%2);
        uint l = min(n, (block_idx  +dir_to_neighbor)*k);
        uint r = min(n, (block_idx+1+dir_to_neighbor)*k);
        while (l<r) {
            uint m = l + (r-l)/2;
            if (code_from[m]<self_code_value + (block_idx&1)) {
                l = m+1;
            } else {
                r = m;
            }
        }
        uint t = gid + l-min(n, (block_idx  +dir_to_neighbor)*k)-k*(block_idx%2);
        code_to[t] = self_code_value;
        indx_to[t] = self_indx_value;
    }
}
