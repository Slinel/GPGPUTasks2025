#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"


#include "../shared_structs/camera_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/aabb_gpu_shared.h"

#include "camera_helpers.cl"
#include "geometry_helpers.cl"
#include "random_helpers.cl"

#define INF 100000.0
#define NINF -100000.0


__attribute__((reqd_work_group_size(32, 1, 1)))
__kernel void reduction(
    __global const float* min_from,
    __global const float* max_from,
    __global       float* min_to,
    __global       float* max_to,
             const uint   k)
{
    __local float mins[32*3];
    __local float maxs[32*3];

    uint gid = get_global_id(0);
    uint guid= get_group_id(0);
    uint lid = get_local_id(0);
    for (int t = 0; t<=64; t+=32) {
        if(guid*32*3+lid+t>=k*3) {
            mins[lid+t] = INF;
            maxs[lid+t] = NINF;
        } else {
            mins[lid+t] = min_from[guid*32*3+lid+t];
            maxs[lid+t] = max_from[guid*32*3+lid+t];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if(lid == 0) {
        float3 Vmax;
        Vmax.x = NINF;
        Vmax.y = NINF;
        Vmax.z = NINF;
        float3 Vmin;
        Vmin.x = INF;
        Vmin.y = INF;
        Vmin.z = INF;
        for(uint i = 0; i<32; ++i) {
            Vmin.x = fmin(Vmin.x, mins[3*i+0]);
            Vmin.y = fmin(Vmin.y, mins[3*i+1]);
            Vmin.z = fmin(Vmin.z, mins[3*i+2]);
            Vmax.x = fmax(Vmax.x, maxs[3*i+0]);
            Vmax.y = fmax(Vmax.y, maxs[3*i+1]);
            Vmax.z = fmax(Vmax.z, maxs[3*i+2]);
        }
        min_to[guid*3+0] = Vmin.x;
        min_to[guid*3+1] = Vmin.y;
        min_to[guid*3+2] = Vmin.z;
        max_to[guid*3+0] = Vmax.x;
        max_to[guid*3+1] = Vmax.y;
        max_to[guid*3+2] = Vmax.z;
        //printf("%d-th group filled the thing\n", guid);
    }
}