#ifndef CUDA_TRIM_H
#define CUDA_TRIM_H

#ifdef __cplusplus
extern "C" {
#endif

// GPU-accelerated head/tail trimming
// Removes front and tail bases from sequences
int gpu_trim_head_tail(
    const char** h_sequences,
    const int* h_seq_lengths,
    int front,
    int tail,
    char** h_output_sequences,
    int* h_output_lengths,
    int num_reads
);

// GPU-accelerated poly-G trimming
// Finds and removes leading/trailing poly-G regions
int gpu_trim_poly_g(
    const char** h_sequences,
    const int* h_seq_lengths,
    int min_g_length,
    int* h_trim_start,
    int* h_trim_end,
    int num_reads
);

// GPU-accelerated quality-based trimming
// Removes low-quality regions using sliding window
int gpu_trim_quality(
    const char** h_sequences,
    const char** h_qualities,
    const int* h_seq_lengths,
    int window_size,
    int quality_threshold,
    int* h_trim_start,
    int* h_trim_end,
    int num_reads
);

#ifdef __cplusplus
}
#endif

#endif // CUDA_TRIM_H
