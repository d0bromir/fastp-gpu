/*
 * cuda_error_check.h — Shared CUDA / nvCOMP error-check macros.
 *
 * Several CUDA translation units (cuda_gzip.cu, gds_pipeline.cu, …) used to
 * define their own CUDA_CHECK/NVCOMP_CHECK macros that differed only in the
 * log tag prefix. Defining the macros once here removes that duplication.
 *
 * Usage:
 *   #define FASTP_CUDA_TAG "my_module"   // optional; default "cuda"
 *   #include "cuda_error_check.h"
 *
 *   FASTP_CUDA_CHECK(cudaMalloc(...));
 *   FASTP_NVCOMP_CHECK(nvcompBatched...Async(...));
 *
 * Both macros log to stderr and `return -1` on failure.
 */
#ifndef FASTP_CUDA_ERROR_CHECK_H
#define FASTP_CUDA_ERROR_CHECK_H

#include <cstdio>
#include <cuda_runtime.h>

#ifndef FASTP_CUDA_TAG
#define FASTP_CUDA_TAG "cuda"
#endif

#define FASTP_CUDA_CHECK(call)                                              \
    do {                                                                    \
        cudaError_t _fastp_e = (call);                                      \
        if (_fastp_e != cudaSuccess) {                                      \
            fprintf(stderr,                                                 \
                    "[" FASTP_CUDA_TAG "] CUDA error %d (%s) at %s:%d\n",   \
                    (int)_fastp_e, cudaGetErrorString(_fastp_e),            \
                    __FILE__, __LINE__);                                    \
            return -1;                                                      \
        }                                                                   \
    } while (0)

#ifdef HAVE_NVCOMP
#include <nvcomp.h>
#define FASTP_NVCOMP_CHECK(call)                                            \
    do {                                                                    \
        nvcompStatus_t _fastp_s = (call);                                   \
        if (_fastp_s != nvcompSuccess) {                                    \
            fprintf(stderr,                                                 \
                    "[" FASTP_CUDA_TAG "] nvCOMP error %d at %s:%d\n",      \
                    (int)_fastp_s, __FILE__, __LINE__);                     \
            return -1;                                                      \
        }                                                                   \
    } while (0)
#endif

#endif // FASTP_CUDA_ERROR_CHECK_H
