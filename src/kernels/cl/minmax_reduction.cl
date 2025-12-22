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



__attribute__((reqd_work_group_size(32, 1, 1)))
__kernel void minmax_reduction(
    __global const float*  vertices,
    __global const uint*   faces,
    __global       float*  to1,
    __global       float*  to2,
                   uint    nfaces)
{
    uint gid = get_global_id(0);
    if(gid>=nfaces) {return;}

    uint3 f = loadFace(faces, gid);
    float3 v0 = loadVertex(vertices, f.x);
    float3 v1 = loadVertex(vertices, f.y);
    float3 v2 = loadVertex(vertices, f.z);

    float3 c = {0.0,0.0,0.0};
    c.x = (v0.x + v1.x + v2.x) * (1.0f / 3.0f);
    c.y = (v0.y + v1.y + v2.y) * (1.0f / 3.0f);
    c.z = (v0.z + v1.z + v2.z) * (1.0f / 3.0f);

    to1[gid*3+0] = c.x;
    to1[gid*3+1] = c.y;
    to1[gid*3+2] = c.z;
    to2[gid*3+0] = c.x;
    to2[gid*3+1] = c.y;
    to2[gid*3+2] = c.z;
}