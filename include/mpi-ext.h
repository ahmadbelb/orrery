#pragma once
#ifndef MPIX_CUDA_AWARE_SUPPORT
#define MPIX_CUDA_AWARE_SUPPORT 1
#endif

#ifdef __cplusplus
extern "C"
#endif
int MPIX_Query_cuda_support(void);
