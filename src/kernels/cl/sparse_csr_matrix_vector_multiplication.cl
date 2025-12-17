#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const uint* csr_row_offsets,
    __global const uint* csr_cols,
    __global const uint* csr_vals,
    __global const uint* vector,
    __global       uint* outp,
                   uint  ncols)
{
    __local uint accum[GROUP_SIZE];
    uint lid = get_local_id(0);
    accum[lid] = 0;

    uint group_id = get_group_id(0);
    uint l = csr_row_offsets[group_id];
    uint r = csr_row_offsets[group_id + 1];

    uint idx = l+lid;
    while (idx<r) {
        uint v = vector[csr_cols[idx]];
        uint m = csr_vals[idx];
        accum[lid]+=(v*m);
        idx+=GROUP_SIZE;
    }

    uint sum = 0;
    if(get_local_id(0)==0) {
        for(uint i = 0; i<GROUP_SIZE; ++i) {
            sum+=accum[i];
        }
        outp[get_group_id(0)] = sum;
    }
}
