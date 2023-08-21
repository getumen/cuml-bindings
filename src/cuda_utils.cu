#include "cuda_utils.h"

namespace cuml4c
{

    __host__ int currentDevice()
    {
        int dev_id;
        CUDA_RT_CALL(cudaGetDevice(&dev_id));
        return dev_id;
    }

} // namespace cuml4c
