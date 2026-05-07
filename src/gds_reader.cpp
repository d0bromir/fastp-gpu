/*
 * GdsReader implementation using the cuFile API.
 *
 * Compiled only when both HAVE_CUDA and HAVE_GDS are defined.
 * When those macros are absent, the no-op stub in gds_reader.h is used
 * instead — no object is produced from this file.
 */

#include "gds_reader.h"

#if defined(HAVE_CUDA) && defined(HAVE_GDS)

#include <fcntl.h>      /* open(), O_RDONLY, O_DIRECT */
#include <unistd.h>     /* close() */
#include <cstdio>       /* fprintf, stderr */
#include <fstream>      /* ifstream for /proc/modules check */
#include <string>
#include <cuda_runtime.h>
#include <cufile.h>

/* ----------------------------------------------------------------------- */
/* Static helper                                                            */
/* ----------------------------------------------------------------------- */

bool GdsReader::isAvailable() {
    /* Verify nvidia-fs kernel module is loaded first */
    {
        std::ifstream modules("/proc/modules");
        if (modules.is_open()) {
            std::string line;
            bool found = false;
            while (std::getline(modules, line)) {
                if (line.compare(0, 9, "nvidia_fs") == 0 ||
                    line.compare(0, 9, "nvidia-fs") == 0) {
                    found = true;
                    break;
                }
            }
            if (!found) return false;
        }
    }
    CUfileError_t st = cuFileDriverOpen();
    if (st.err != CU_FILE_SUCCESS) return false;
    cuFileDriverClose();
    return true;
}

/* ----------------------------------------------------------------------- */
/* Constructor / Destructor                                                 */
/* ----------------------------------------------------------------------- */

GdsReader::GdsReader(int device)
    : mDevice(device), mFd(-1), mOpen(false), mDriverOpen(false)
{
    cudaSetDevice(mDevice);
    CUfileError_t st = cuFileDriverOpen();
    mDriverOpen = (st.err == CU_FILE_SUCCESS);
    if (!mDriverOpen)
        fprintf(stderr,
                "GdsReader: cuFileDriverOpen failed (err=%d); "
                "GPU-Direct Storage reads will not be available.\n",
                (int)st.err);
}

GdsReader::~GdsReader() {
    close();
    if (mDriverOpen) cuFileDriverClose();
}

/* ----------------------------------------------------------------------- */
/* File management                                                          */
/* ----------------------------------------------------------------------- */

int GdsReader::open(const char* path) {
    if (!mDriverOpen) return -1;
    if (mOpen) close();   /* close any previously opened file */

    mFd = ::open(path, O_RDONLY | O_DIRECT);
    if (mFd < 0) {
        fprintf(stderr, "GdsReader::open: cannot open '%s'\n", path);
        return -1;
    }

    CUfileDescr_t desc{};
    desc.handle.fd = mFd;
    desc.type      = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    CUfileError_t st = cuFileHandleRegister(&mHandle, &desc);
    if (st.err != CU_FILE_SUCCESS) {
        fprintf(stderr,
                "GdsReader::open: cuFileHandleRegister failed (err=%d)\n",
                (int)st.err);
        ::close(mFd);
        mFd = -1;
        return -1;
    }

    mOpen = true;
    return 0;
}

void GdsReader::close() {
    if (mOpen) {
        cuFileHandleDeregister(mHandle);
        ::close(mFd);
        mFd  = -1;
        mOpen = false;
    }
}

/* ----------------------------------------------------------------------- */
/* Buffer registration                                                      */
/* ----------------------------------------------------------------------- */

int GdsReader::registerBuffer(void* d_buf, size_t bytes) {
    CUfileError_t st = cuFileBufRegister(d_buf, bytes, 0);
    if (st.err != CU_FILE_SUCCESS) {
        fprintf(stderr,
                "GdsReader::registerBuffer: cuFileBufRegister failed (err=%d)\n",
                (int)st.err);
        return -1;
    }
    return 0;
}

void GdsReader::deregisterBuffer(void* d_buf) {
    cuFileBufDeregister(d_buf);
}

/* ----------------------------------------------------------------------- */
/* Direct NVMe -> GPU read                                                  */
/* ----------------------------------------------------------------------- */

ssize_t GdsReader::read(void* d_buf, size_t count, off_t file_offset) {
    if (!mOpen) return -1;

    /* cuFileRead signature:
     *   ssize_t cuFileRead(CUfileHandle_t fh, void* buf_ptr_or_offset,
     *                      size_t size, off_t file_offset, off_t buf_offset)
     * buf_offset = 0: write to the start of the registered buffer. */
    ssize_t bytes = cuFileRead(mHandle, d_buf, count,
                               (off_t)file_offset, (off_t)0);
    if (bytes < 0) {
        fprintf(stderr,
                "GdsReader::read: cuFileRead failed (ret=%zd)\n", bytes);
        return -1;
    }
    return bytes;
}

#endif /* HAVE_CUDA && HAVE_GDS */
