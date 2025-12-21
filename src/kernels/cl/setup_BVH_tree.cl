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




static inline int clz32(uint x) {
    if (x == 0u) return 32;
    int n = 0;
    if (x <= 0x0000FFFF) { n += 16; x <<= 16; }
    if (x <= 0x00FFFFFF) { n += 8;  x <<= 8; }
    if (x <= 0x0FFFFFFF) { n += 4;  x <<= 4; }
    if (x <= 0x3FFFFFFF) { n += 2;  x <<= 2; }
    if (x <= 0x7FFFFFFF) { n += 1; }
    return n;
}

static inline int common_prefix(__global const MortonCode* codes, int N, int i, int j)
{
    if (j < 0 || j >= N) return -1;

    MortonCode ci = codes[(uint)(i)];
    MortonCode cj = codes[(uint)(j)];

    if (ci == cj) {
        uint di = (uint)(i);
        uint dj = (uint)(j);
        uint diff = di ^ dj;
        return 32 + clz32(diff);
    } else {
        uint diff = ci ^ cj;
        return clz32(diff);
    }
}


// Determine range [first, last] of primitives covered by internal node i
static inline void determine_range(__global const MortonCode* codes, int N, int i, int* outFirst, int* outLast)
{
    int cpL = common_prefix(codes, N, i, i - 1);
    int cpR = common_prefix(codes, N, i, i + 1);

    // Direction of the range
    int d = (cpR > cpL) ? 1 : -1;

    // Find upper bound on the length of the range
    int deltaMin = common_prefix(codes, N, i, i - d);
    int lmax = 2;

    while (common_prefix(codes, N, i, i + lmax * d) > deltaMin) {
        lmax <<= 1;
    }

    // Binary search to find exact range length
    int l = 0;
    for (int t = lmax >> 1; t > 0; t >>= 1) {
        if (common_prefix(codes, N, i, i + (l + t) * d) > deltaMin) {
            l += t;
        }
    }

    int j = i + l * d;
    *outFirst = min(i, j);
    *outLast  = max(i, j);
}

// Find split position inside range [first, last] using the same
// prefix metric as determine_range (code + index tie-break)
static inline int find_split(__global const MortonCode* codes, int first, int last, int nfaces)
{
    const int N = (int)(nfaces);

    // Degenerate case should not случаться в нормальном коде, но на всякий пожарный
    if (first == last)
        return first;


    // Prefix between first and last (уже с учётом индекса, если коды равны)
    int commonPrefix = common_prefix(codes, N, first, last);

    int split = first;
    int step  = last - first;

    // Binary search for the last index < last where
    // prefix(first, i) > prefix(first, last)
    do {
        step = (step + 1) >> 1;
        int newSplit = split + step;

        if (newSplit < last) {
            int splitPrefix = common_prefix(codes, N, first, newSplit);
            if (splitPrefix > commonPrefix) {
                split = newSplit;
            }
        }
    } while (step > 1);

    return split;
}


__attribute__((reqd_work_group_size(32, 1, 1)))
__kernel void setup_BVH_tree(
    __global const MortonCode* sortedCodes,
    __global       BVHNodeGPU* outNodes,
    __global       uint*       parents,
    uint                       nfaces)
{
    uint gid = get_global_id(0);
    int i = (int)gid;

    if(0 <= i && i < (nfaces-1)) {
        int first, last;
        determine_range(sortedCodes, (int)(nfaces), i, &first, &last);
        int split = find_split(sortedCodes, first, last, nfaces);

        // Left child
        int leftIndex;
        if (split == first) {
            // Range [first, split] has one primitive -> leaf
            leftIndex = (int)((nfaces - 1) + split);
        } else {
            // Internal node
            leftIndex = split;
        }

        // Right child
        int rightIndex;
        if (split + 1 == last) {
            // Range [split+1, last] has one primitive -> leaf
            rightIndex = (int)((nfaces - 1) + split + 1);
        } else {
            // Internal node
            rightIndex = split + 1;
        }

        outNodes[(uint)(i)].leftChildIndex  = (uint)(leftIndex);
        outNodes[(uint)(i)].rightChildIndex = (uint)(rightIndex);
        outNodes[(uint)(i)].aabb.min_x = 0.0; //надо чтобы атомарные флаги работали
        parents[(uint)(leftIndex) ] = (uint)(i);
        parents[(uint)(rightIndex)] = (uint)(i);
    }
}
