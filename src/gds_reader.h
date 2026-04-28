#pragma once
/*
 * GPU-Direct Storage (GDS) reader — Future Direction 3
 *
 * Wraps the cuFile API to provide direct NVMe-to-GPU DMA transfers,
 * bypassing the CPU and host-memory bounce buffer entirely.
 *
 * Requirements:
 *   - CUDA 11.4+ (cuFile shipped alongside libcufile.so)
 *   - NVIDIA GDS-capable kernel driver (see `gds_check` utility)
 *   - Hardware that supports P2P DMA (most server-grade NICs and NVMe)
 *   - Files on a GDS-capable filesystem (NVMe, Lustre, GPFS, WEKA, etc.)
 *
 * Build with:
 *   make WITH_CUDA=1 WITH_GDS=1
 *
 * When HAVE_CUDA or HAVE_GDS is not defined, a no-op stub class is
 * compiled instead so the rest of the code can call GdsReader methods
 * unconditionally without #ifdef guards at the call sites.
 *
 * Typical usage:
 *   GdsReader reader(device_id);
 *   if (GdsReader::isAvailable() && reader.valid()) {
 *       reader.open("/path/to/input.fastq.bgz");
 *       reader.registerBuffer(d_buf, buf_bytes);
 *       ssize_t n = reader.read(d_buf, buf_bytes, file_offset);
 *       // d_buf now contains data loaded directly from NVMe into GPU memory
 *       reader.deregisterBuffer(d_buf);
 *       reader.close();
 *   }
 */

#if defined(HAVE_CUDA) && defined(HAVE_GDS)

#include <cuda_runtime.h>
#include <cufile.h>
#include <cstddef>
#include <sys/types.h>   /* ssize_t, off_t */

class GdsReader {
public:
    /** Initialise the cuFile driver for the given CUDA device. */
    explicit GdsReader(int device = 0);
    ~GdsReader();

    /** Returns true if the cuFile driver can be opened on this system. */
    static bool isAvailable();

    /**
     * Open a file for GDS reads.
     * The file is opened O_RDONLY | O_DIRECT and registered with cuFile.
     * @param path  Path to the file.
     * @return 0 on success, -1 on error.
     */
    int open(const char* path);

    /** Close the file and release cuFile resources. */
    void close();

    /**
     * Register a device buffer for GDS DMA.
     * Must be called once per buffer before the first read() using it.
     * @param d_buf   cudaMalloc'd device pointer.
     * @param bytes   Size in bytes.
     * @return 0 on success, -1 on error.
     */
    int registerBuffer(void* d_buf, size_t bytes);

    /** Deregister a previously registered device buffer. */
    void deregisterBuffer(void* d_buf);

    /**
     * Read directly from file into GPU memory (NVMe -> GPU DMA).
     * The destination buffer must have been registered via registerBuffer().
     * @param d_buf        Registered device pointer.
     * @param count        Bytes to read.
     * @param file_offset  Byte offset within the file.
     * @return Bytes transferred (>= 0), or -1 on error.
     */
    ssize_t read(void* d_buf, size_t count, off_t file_offset);

    /** True if the driver opened successfully and a file is registered. */
    bool valid() const { return mDriverOpen; }

private:
    int            mDevice;
    int            mFd;
    bool           mOpen;
    bool           mDriverOpen;
    CUfileHandle_t mHandle;
};

#else  /* ---- No-op stub when HAVE_CUDA or HAVE_GDS is absent ---- */

#include <cstddef>    /* size_t  */
#include <sys/types.h> /* ssize_t, off_t */

class GdsReader {
public:
    explicit GdsReader(int = 0) {}
    ~GdsReader() = default;

    static bool isAvailable()               { return false; }
    int    open(const char*)                { return -1; }
    void   close()                          {}
    int    registerBuffer(void*, size_t)    { return -1; }
    void   deregisterBuffer(void*)          {}
    ssize_t read(void*, size_t, off_t)      { return -1; }
    bool   valid() const                    { return false; }
};

#endif /* HAVE_CUDA && HAVE_GDS */
