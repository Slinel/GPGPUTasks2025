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

// Взято с cpu версии
static inline unsigned int expandBits(unsigned int v)
{
    // Ensure we have only lowest 10 bits
    //rassert(v == (v & 0x3FFu), 76389413321, v);

    // Magic bit expansion steps
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;

    return v;
}

static inline MortonCode morton3D(float x, float y, float z)
{
    // Map and clamp to integer grid [0, 1023]
    unsigned int ix = min(max((int) (x * 1024.0f), 0), 1023);
    unsigned int iy = min(max((int) (y * 1024.0f), 0), 1023);
    unsigned int iz = min(max((int) (z * 1024.0f), 0), 1023);

    unsigned int xx = expandBits(ix);
    unsigned int yy = expandBits(iy);
    unsigned int zz = expandBits(iz);

    // Interleave: x in bits [2,5,8,...], y in [1,4,7,...], z in [0,3,6,...]
    return (xx << 2) | (yy << 1) | zz;
}

__attribute__((reqd_work_group_size(32, 1, 1)))
__kernel void build_morton(
    __global const float*      vertices,
    __global const uint*       faces,
    __global       MortonCode* codes,
    __global       uint*       leaf_faces_indices_gpu,
    uint                       nfaces)
{
    uint gid = get_global_id(0);
    if (gid >= nfaces) return;
    leaf_faces_indices_gpu[gid] = gid;

    uint3 face = loadFace(faces, gid);
    float3 v0 = loadVertex(vertices, face.x);
    float3 v1 = loadVertex(vertices, face.y);
    float3 v2 = loadVertex(vertices, face.z);

    float3 centroid;
    centroid.x = (v0.x + v1.x + v2.x) * (1.0f / 3.0f);
    centroid.y = (v0.y + v1.y + v2.y) * (1.0f / 3.0f);
    centroid.z = (v0.z + v1.z + v2.z) * (1.0f / 3.0f);

    codes[gid] = morton3D(
        centroid.x,
        centroid.y,
        centroid.z
    );
}
