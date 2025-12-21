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
__kernel void denoise(
    __global const float*      from,
    __global       float*      to,
    uint                       width,
    uint                       height)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);
    if(gidx>=width) return;
    if(gidy>=height) return;
    float accum = 0;
    int count = 0;
    for(int dy = -9; dy<=9; ++dy) {
        for(int dx = -9; dx<=9; ++dx) {
            if( 0 > (dx+gidx) || (dx+gidx) >= width || 0 > (dy+gidy) || (dy+gidy) >= height) {
                continue;
            } else {
                float r2 = -(dx*dx + dy*dy);
                accum+= from[(dx+gidx) + width*(dy+gidy)] * exp(r2/8);
                //accum+= from[(dx+gidx) + width*(dy+gidy)];
                //count++;
            }
        }
    }
    to[gidx + width*gidy] = accum;
}
