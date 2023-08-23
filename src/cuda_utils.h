#pragma once

#include <cstdio>
#include <stdexcept>
#include <cuda_runtime.h>

#ifndef NORET
#if defined(__GNUC__) && __GNUC__ >= 3
#define NORET __attribute__((noreturn))
#else
#define NORET
#endif
#endif

#ifndef cudaEventWaitDefault
#define cudaEventWaitDefault 0x00
#endif

#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL(call)                                                               \
    {                                                                                    \
        auto const cudaStatus = (call);                                                  \
        if (cudaSuccess != cudaStatus)                                                   \
        {                                                                                \
            cuml4c::stop("ERROR: CUDA RT call \"%s\" in line %d of file %s failed with " \
                         "%s (%d).\n",                                                   \
                         #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus),      \
                         cudaStatus);                                                    \
        }                                                                                \
    }
#endif

namespace cuml4c
{

    int currentDevice();

    template <typename... Args>
    std::string format(const std::string &fmt, Args... args)
    {
        size_t len = std::snprintf(nullptr, 0, fmt.c_str(), args...);
        std::vector<char> buf(len + 1);
        std::snprintf(&buf[0], len + 1, fmt.c_str(), args...);
        return std::string(&buf[0], &buf[0] + len);
    }

    template <typename... Args>
    void NORET stop(const char *fmt, Args... args)
    {
        throw std::runtime_error(format(fmt, args...));
    }

} // namespace cuml4c
