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
static inline unsigned int expandBits(unsigned int v) {
    v = v & 0x3FFu;
    v = (v | (v << 16)) & 0x030000FFu;
    v = (v | (v << 8))  & 0x0300F00Fu;
    v = (v | (v << 4))  & 0x030C30C3u;
    v = (v | (v << 2))  & 0x09249249u;
    return v;
}

static inline MortonCode morton3D(float x, float y, float z) {
    unsigned int ix = min(max((int) (x * 1024.0f), 0), 1023);
    unsigned int iy = min(max((int) (y * 1024.0f), 0), 1023);
    unsigned int iz = min(max((int) (z * 1024.0f), 0), 1023);
    unsigned int xx = expandBits(ix);
    unsigned int yy = expandBits(iy);
    unsigned int zz = expandBits(iz);
    return (xx << 2) | (yy << 1) | zz;
}

__attribute__((reqd_work_group_size(32, 1, 1)))
__kernel void build_morton(
    __global const float*      vertices,
    __global const uint*       faces,
    __global       MortonCode* codes,
    __global       uint*       leaf_faces_indices_gpu,
             const uint        nfaces,
             const float      cMinx,
             const float      cMiny,
             const float      cMinz,
             const float      cMaxx,
             const float      cMaxy,
             const float      cMaxz)
{
        uint gid = get_global_id(0);
        if(gid>=nfaces) {return;}
        leaf_faces_indices_gpu[gid] = gid;
        int i = (int)gid;

        const float eps = 1e-9f;
        const float dx = max(cMaxx - cMinx, eps);
        const float dy = max(cMaxy - cMiny, eps);
        const float dz = max(cMaxz - cMinz, eps);

        const uint3 f = loadFace(faces, i);
        const float3 v0 = loadVertex(vertices, f.x);
        const float3 v1 = loadVertex(vertices, f.y);
        const float3 v2 = loadVertex(vertices, f.z);

        float3 c;
        c.x = (v0.x + v1.x + v2.x) * (1.0f / 3.0f);
        c.y = (v0.y + v1.y + v2.y) * (1.0f / 3.0f);
        c.z = (v0.z + v1.z + v2.z) * (1.0f / 3.0f);

        float nx = (c.x - cMinx) / dx;
        float ny = (c.y - cMiny) / dy;
        float nz = (c.z - cMinz) / dz;

        nx = min(max(nx, 0.0f), 1.0f);
        ny = min(max(ny, 0.0f), 1.0f);
        nz = min(max(nz, 0.0f), 1.0f);

        codes[i] = morton3D(nx, ny, nz);
}
