#include "cuda_trim.h"
#include <stdio.h>
#include <string.h>

// GPU kernel for head/tail trimming
// Removes front and tail bases from sequences
__global__ void trim_head_tail_kernel(
    char** d_sequences,
    int* d_seq_lengths,
    int front,
    int tail,
    char** d_output_sequences,
    int* d_output_lengths,
    int num_reads
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= num_reads) return;

    int orig_len = d_seq_lengths[idx];
    int new_len = orig_len - front - tail;
    
    if(new_len <= 0) {
        d_output_lengths[idx] = 0;
        return;
    }

    d_output_lengths[idx] = new_len;
    
    // Copy trimmed sequence
    char* src = d_sequences[idx];
    char* dst = d_output_sequences[idx];
    
    for(int i = threadIdx.x; i < new_len; i += blockDim.x) {
        dst[i] = src[front + i];
    }
}

// GPU kernel for poly-G trimming
// Finds and removes leading/trailing poly-G regions
__global__ void trim_poly_g_kernel(
    char** d_sequences,
    int* d_seq_lengths,
    int min_g_length,
    int* d_trim_start,
    int* d_trim_end,
    int num_reads
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= num_reads) return;

    char* seq = d_sequences[idx];
    int len = d_seq_lengths[idx];
    int trim_start = 0;
    int trim_end = len;

    // Find leading poly-G
    int g_count = 0;
    for(int i = 0; i < len; i++) {
        if(seq[i] == 'G' || seq[i] == 'g') {
            g_count++;
            if(g_count >= min_g_length) {
                trim_start = i + 1;
                break;
            }
        } else {
            g_count = 0;
        }
    }

    // Find trailing poly-G
    g_count = 0;
    for(int i = len - 1; i >= trim_start; i--) {
        if(seq[i] == 'G' || seq[i] == 'g') {
            g_count++;
            if(g_count >= min_g_length) {
                trim_end = i;
                break;
            }
        } else {
            g_count = 0;
        }
    }

    d_trim_start[idx] = trim_start;
    d_trim_end[idx] = (trim_end > trim_start) ? trim_end : trim_start;
}

// GPU kernel for quality-based trimming with sliding window
// Removes low-quality regions using a sliding window approach
__global__ void trim_quality_kernel(
    char** d_sequences,
    char** d_qualities,
    int* d_seq_lengths,
    int window_size,
    int quality_threshold,
    int* d_trim_start,
    int* d_trim_end,
    int num_reads
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= num_reads) return;

    char* qual = d_qualities[idx];
    int len = d_seq_lengths[idx];
    
    int trim_start = 0;
    int trim_end = len;

    if(len < window_size) {
        d_trim_start[idx] = 0;
        d_trim_end[idx] = len;
        return;
    }

    // Find start position (forward scan)
    int total_qual = 0;
    for(int i = 0; i < window_size - 1; i++) {
        total_qual += (qual[i] - 33);
    }

    for(int i = 0; i + window_size < len; i++) {
        total_qual += (qual[i + window_size - 1] - 33);
        if(i > 0) {
            total_qual -= (qual[i - 1] - 33);
        }
        
        int avg_qual = total_qual / window_size;
        if(avg_qual >= quality_threshold) {
            trim_start = i;
            break;
        }
    }

    // Find end position (reverse scan)
    total_qual = 0;
    for(int i = len - window_size; i < len - 1; i++) {
        total_qual += (qual[i] - 33);
    }

    for(int i = len - 1; i >= trim_start + window_size; i--) {
        total_qual += (qual[i - window_size] - 33);
        if(i < len - 1) {
            total_qual -= (qual[i + 1] - 33);
        }
        
        int avg_qual = total_qual / window_size;
        if(avg_qual >= quality_threshold) {
            trim_end = i + 1;
            break;
        }
    }

    d_trim_start[idx] = trim_start;
    d_trim_end[idx] = (trim_end > trim_start) ? trim_end : trim_start;
}

// Host function: Head/tail trimming
int gpu_trim_head_tail(
    const char** h_sequences,
    const int* h_seq_lengths,
    int front,
    int tail,
    char** h_output_sequences,
    int* h_output_lengths,
    int num_reads
) {
    if(num_reads <= 0) return 0;

    // Allocate device memory for sequence pointers and lengths
    char** d_sequences = NULL;
    char** d_output_sequences = NULL;
    int* d_seq_lengths = NULL;
    int* d_output_lengths = NULL;

    cudaMalloc(&d_sequences, num_reads * sizeof(char*));
    cudaMalloc(&d_output_sequences, num_reads * sizeof(char*));
    cudaMalloc(&d_seq_lengths, num_reads * sizeof(int));
    cudaMalloc(&d_output_lengths, num_reads * sizeof(int));

    // Copy metadata
    cudaMemcpy(d_seq_lengths, h_seq_lengths, num_reads * sizeof(int), cudaMemcpyHostToDevice);

    // Copy sequence pointers (addresses on device won't work, need special handling)
    // For now, allocate contiguous buffer and reorganize
    // This is a simplified version - in production, use pinned memory + device buffer
    
    for(int i = 0; i < num_reads; i++) {
        char* d_seq = NULL;
        int seq_len = h_seq_lengths[i];
        if(seq_len > 0) {
            cudaMalloc(&d_seq, seq_len + 1);
            cudaMemcpy(d_seq, h_sequences[i], seq_len, cudaMemcpyHostToDevice);
        }
        cudaMemcpy(d_sequences + i, &d_seq, sizeof(char*), cudaMemcpyHostToDevice);
    }

    // Allocate output sequences
    for(int i = 0; i < num_reads; i++) {
        int new_len = h_seq_lengths[i] - front - tail;
        if(new_len > 0) {
            char* d_out = NULL;
            cudaMalloc(&d_out, new_len + 1);
            cudaMemcpy(d_output_sequences + i, &d_out, sizeof(char*), cudaMemcpyHostToDevice);
        }
    }

    // Launch kernel
    int block_size = 256;
    int grid_size = (num_reads + block_size - 1) / block_size;
    trim_head_tail_kernel<<<grid_size, block_size>>>(
        d_sequences, d_seq_lengths, front, tail,
        d_output_sequences, d_output_lengths, num_reads
    );

    // Copy results back
    cudaMemcpy(h_output_lengths, d_output_lengths, num_reads * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i < num_reads; i++) {
        if(h_output_lengths[i] > 0) {
            char* d_out = NULL;
            cudaMemcpy(&d_out, d_output_sequences + i, sizeof(char*), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_output_sequences[i], d_out, h_output_lengths[i], cudaMemcpyDeviceToHost);
            h_output_sequences[i][h_output_lengths[i]] = '\0';
            cudaFree(d_out);
        }
    }

    // Cleanup device memory
    for(int i = 0; i < num_reads; i++) {
        char* d_seq = NULL;
        cudaMemcpy(&d_seq, d_sequences + i, sizeof(char*), cudaMemcpyDeviceToHost);
        if(d_seq) cudaFree(d_seq);
    }
    cudaFree(d_sequences);
    cudaFree(d_output_sequences);
    cudaFree(d_seq_lengths);
    cudaFree(d_output_lengths);

    return 0;
}

// Host function: Poly-G trimming
int gpu_trim_poly_g(
    const char** h_sequences,
    const int* h_seq_lengths,
    int min_g_length,
    int* h_trim_start,
    int* h_trim_end,
    int num_reads
) {
    if(num_reads <= 0) return 0;

    char** d_sequences = NULL;
    int* d_seq_lengths = NULL;
    int* d_trim_start = NULL;
    int* d_trim_end = NULL;

    cudaMalloc(&d_sequences, num_reads * sizeof(char*));
    cudaMalloc(&d_seq_lengths, num_reads * sizeof(int));
    cudaMalloc(&d_trim_start, num_reads * sizeof(int));
    cudaMalloc(&d_trim_end, num_reads * sizeof(int));

    cudaMemcpy(d_seq_lengths, h_seq_lengths, num_reads * sizeof(int), cudaMemcpyHostToDevice);

    // Copy sequence pointers to device
    for(int i = 0; i < num_reads; i++) {
        char* d_seq = NULL;
        int seq_len = h_seq_lengths[i];
        if(seq_len > 0) {
            cudaMalloc(&d_seq, seq_len + 1);
            cudaMemcpy(d_seq, h_sequences[i], seq_len, cudaMemcpyHostToDevice);
        }
        cudaMemcpy(d_sequences + i, &d_seq, sizeof(char*), cudaMemcpyHostToDevice);
    }

    // Launch kernel
    int block_size = 256;
    int grid_size = (num_reads + block_size - 1) / block_size;
    trim_poly_g_kernel<<<grid_size, block_size>>>(
        d_sequences, d_seq_lengths, min_g_length,
        d_trim_start, d_trim_end, num_reads
    );

    // Copy results back
    cudaMemcpy(h_trim_start, d_trim_start, num_reads * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_trim_end, d_trim_end, num_reads * sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup
    for(int i = 0; i < num_reads; i++) {
        char* d_seq = NULL;
        cudaMemcpy(&d_seq, d_sequences + i, sizeof(char*), cudaMemcpyDeviceToHost);
        if(d_seq) cudaFree(d_seq);
    }
    cudaFree(d_sequences);
    cudaFree(d_seq_lengths);
    cudaFree(d_trim_start);
    cudaFree(d_trim_end);

    return 0;
}

// Host function: Quality-based trimming
int gpu_trim_quality(
    const char** h_sequences,
    const char** h_qualities,
    const int* h_seq_lengths,
    int window_size,
    int quality_threshold,
    int* h_trim_start,
    int* h_trim_end,
    int num_reads
) {
    if(num_reads <= 0) return 0;

    char** d_sequences = NULL;
    char** d_qualities = NULL;
    int* d_seq_lengths = NULL;
    int* d_trim_start = NULL;
    int* d_trim_end = NULL;

    cudaMalloc(&d_sequences, num_reads * sizeof(char*));
    cudaMalloc(&d_qualities, num_reads * sizeof(char*));
    cudaMalloc(&d_seq_lengths, num_reads * sizeof(int));
    cudaMalloc(&d_trim_start, num_reads * sizeof(int));
    cudaMalloc(&d_trim_end, num_reads * sizeof(int));

    cudaMemcpy(d_seq_lengths, h_seq_lengths, num_reads * sizeof(int), cudaMemcpyHostToDevice);

    // Copy sequence and quality pointers to device
    for(int i = 0; i < num_reads; i++) {
        char* d_seq = NULL;
        char* d_qual = NULL;
        int seq_len = h_seq_lengths[i];
        if(seq_len > 0) {
            cudaMalloc(&d_seq, seq_len + 1);
            cudaMalloc(&d_qual, seq_len + 1);
            cudaMemcpy(d_seq, h_sequences[i], seq_len, cudaMemcpyHostToDevice);
            cudaMemcpy(d_qual, h_qualities[i], seq_len, cudaMemcpyHostToDevice);
        }
        cudaMemcpy(d_sequences + i, &d_seq, sizeof(char*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_qualities + i, &d_qual, sizeof(char*), cudaMemcpyHostToDevice);
    }

    // Launch kernel
    int block_size = 256;
    int grid_size = (num_reads + block_size - 1) / block_size;
    trim_quality_kernel<<<grid_size, block_size>>>(
        d_sequences, d_qualities, d_seq_lengths, window_size, quality_threshold,
        d_trim_start, d_trim_end, num_reads
    );

    // Copy results back
    cudaMemcpy(h_trim_start, d_trim_start, num_reads * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_trim_end, d_trim_end, num_reads * sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup
    for(int i = 0; i < num_reads; i++) {
        char* d_seq = NULL;
        char* d_qual = NULL;
        cudaMemcpy(&d_seq, d_sequences + i, sizeof(char*), cudaMemcpyDeviceToHost);
        cudaMemcpy(&d_qual, d_qualities + i, sizeof(char*), cudaMemcpyDeviceToHost);
        if(d_seq) cudaFree(d_seq);
        if(d_qual) cudaFree(d_qual);
    }
    cudaFree(d_sequences);
    cudaFree(d_qualities);
    cudaFree(d_seq_lengths);
    cudaFree(d_trim_start);
    cudaFree(d_trim_end);

    return 0;
}
