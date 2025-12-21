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

#define INVALID 0xFFFFFFFFu

__attribute__((reqd_work_group_size(32, 1, 1)))
__kernel void calculate_aabbs(
    __global       BVHNodeGPU* Nodes,
    __global const uint*       parents,
    __global const uint*       sortedFaceIndices,
    __global const float*      vertices,
    __global const uint*       faces,
    uint                       nfaces)
{
    uint gid = get_global_id(0);
    if(gid>=nfaces) return;

    uint3 face = loadFace(faces, sortedFaceIndices[gid]);
    float3 v0, v1, v2;
    v0 = loadVertex(vertices, face.x);
    v1 = loadVertex(vertices, face.y);
    v2 = loadVertex(vertices, face.z);

    AABBGPU aabb;
    aabb.min_x = fmin(v0.x, fmin(v1.x, v2.x));
    aabb.min_y = fmin(v0.y, fmin(v1.y, v2.y));
    aabb.min_z = fmin(v0.z, fmin(v1.z, v2.z));
    aabb.max_x = fmax(v0.x, fmax(v1.x, v2.x));
    aabb.max_y = fmax(v0.y, fmax(v1.y, v2.y));
    aabb.max_z = fmax(v0.z, fmax(v1.z, v2.z));

    uint leaf = (nfaces - 1) + gid;
    Nodes[leaf].leftChildIndex  = INVALID;
    Nodes[leaf].rightChildIndex = INVALID;
    Nodes[leaf].aabb = aabb;

    do {
        leaf = parents[leaf];
        // наглядный пример того как НЕ надо нарушать типизацию (на чьей-то машине оно точно сломается)
        if ((float)(atomic_cmpxchg((volatile __global uint*)(&(Nodes[leaf].aabb.min_x)), 0, 1))==0.0) {
            break;
        }
        BVHNodeGPU V = Nodes[leaf];
        AABBGPU laabb = Nodes[V.leftChildIndex ].aabb;
        AABBGPU raabb = Nodes[V.rightChildIndex].aabb;
        aabb.min_x = fmin(laabb.min_x, raabb.min_x);
        aabb.min_y = fmin(laabb.min_y, raabb.min_y);
        aabb.min_z = fmin(laabb.min_z, raabb.min_z);
        aabb.max_x = fmax(laabb.max_x, raabb.max_x);
        aabb.max_y = fmax(laabb.max_y, raabb.max_y);
        aabb.max_z = fmax(laabb.max_z, raabb.max_z);
        Nodes[leaf].aabb = aabb;
    } while (leaf!=0);
}
