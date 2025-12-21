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

__attribute__((reqd_work_group_size(32, 1, 1)))
__kernel void setup_BVH_tree(
    __global const MortonCode* code,
    __global const uint*       idx,
    __global       BVHNodeGPU* nodes,
    uint                       nfaces)
{
    uint gid = get_global_id(0);
    // Делаем листья
    if(gid<nfaces) {
        nodes[gid+nfaces-1].leftChildIndex  = INVALID;
        nodes[gid+nfaces-1].RightChildIndex = INVALID;
        nodes[gid+nfaces-1].aabb = {};
    }

    // В этом же кернеле делаем промежуточные ноды
    // Потому что я заколебался создавать новые кернелы, сюда тоже норм влезет
    if(gid<nfaces-1) {
        int first, last;
        determine_range(sortedCodes, static_cast<int>(N), i, first, last);
        int split = find_split(sortedCodes, first, last);

        // Left child
        int leftIndex;
        if (split == first) {
            // Range [first, split] has one primitive -> leaf
            leftIndex = static_cast<int>((N - 1) + split);
        } else {
            // Internal node
            leftIndex = split;
        }

        // Right child
        int rightIndex;
        if (split + 1 == last) {
            // Range [split+1, last] has one primitive -> leaf
            rightIndex = static_cast<int>((N - 1) + split + 1);
        } else {
            // Internal node
            rightIndex = split + 1;
        }

        BVHNodeGPU& node = outNodes[static_cast<size_t>(i)];
        node.leftChildIndex  = static_cast<GPUC_UINT>(leftIndex);
        node.rightChildIndex = static_cast<GPUC_UINT>(rightIndex);
    }
}
