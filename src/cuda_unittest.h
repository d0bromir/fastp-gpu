#ifndef CUDA_UNITTEST_H
#define CUDA_UNITTEST_H

/*
 * cuda_unittest.h — host-side unit tests for the GPU/CUDA code paths added
 * by the d0bromir fork.  Every sub-test is gated by the appropriate
 * HAVE_CUDA / HAVE_NVCOMP / HAVE_GDS macro and additionally auto-skips at
 * runtime when no CUDA device is visible to the process, so the same
 * binary can pass tests on a developer laptop without a GPU.
 *
 * Mirrors the static ::test() pattern used by Sequence, Read, etc., so it
 * can be invoked from UnitTest::run() without conditional compilation in
 * the caller.
 */
class CudaUnitTest {
public:
    /** Run all GPU/CUDA unit tests.  Returns true iff every enabled
     *  sub-test passed (or was legitimately skipped). */
    static bool test();
};

#endif // CUDA_UNITTEST_H
