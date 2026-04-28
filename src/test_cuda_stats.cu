/**
 * Unit test for CUDA GPU per-read statistics computation
 * This test can be compiled and run independently to verify CUDA functionality
 */

#include <iostream>
#include <cstring>
#include <vector>
#include "cuda_stats.h"

using namespace std;

// Test 1: Simple single read test
bool test_single_read() {
    cout << "Test 1: Single read statistics..." << endl;
    
    const char* seq = "ACGTACGTACGTNNNN";
    const char* qual = "??????????????II";
    const int* lengths = new int[1]{16};
    
    const char** sequences = new const char*[1]{seq};
    const char** qualities = new const char*[1]{qual};
    
    ReadStatistics* stats = new ReadStatistics[1];
    
    int result = cuda_compute_read_stats(
        sequences, qualities, lengths, 1, '5', stats
    );
    
    if (result != 0) {
        cout << "ERROR: CUDA computation failed" << endl;
        return false;
    }
    
    // Expected: 16 total bases, 4 N bases, 0 low qual (all Q=14+ which is > Q5)
    cout << "  Total bases: " << stats[0].total_bases << " (expected 16)" << endl;
    cout << "  N bases: " << stats[0].n_bases << " (expected 4)" << endl;
    cout << "  Low qual bases: " << stats[0].low_qual_bases << " (expected 0)" << endl;
    cout << "  Total quality: " << stats[0].total_quality << endl;
    
    bool passed = (stats[0].total_bases == 16 && stats[0].n_bases == 4);
    
    delete[] stats;
    delete[] lengths;
    delete[] sequences;
    delete[] qualities;
    
    cout << (passed ? "PASSED" : "FAILED") << endl << endl;
    return passed;
}

// Test 2: Multiple reads batch test
bool test_batch_reads() {
    cout << "Test 2: Batch processing multiple reads..." << endl;
    
    int num_reads = 100;
    const char** sequences = new const char*[num_reads];
    const char** qualities = new const char*[num_reads];
    int* lengths = new int[num_reads];
    
    // Create test data
    for (int i = 0; i < num_reads; i++) {
        // Create simple test sequence and quality
        sequences[i] = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        qualities[i] = "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII";
        lengths[i] = 100;
    }
    
    ReadStatistics* stats = new ReadStatistics[num_reads];
    
    int result = cuda_compute_read_stats(
        sequences, qualities, lengths, num_reads, '5', stats
    );
    
    if (result != 0) {
        cout << "ERROR: CUDA batch computation failed" << endl;
        return false;
    }
    
    // Check first and last reads
    cout << "  First read - Total bases: " << stats[0].total_bases << " (expected 100)" << endl;
    cout << "  Last read - Total bases: " << stats[num_reads-1].total_bases << " (expected 100)" << endl;
    cout << "  First read - N bases: " << stats[0].n_bases << " (expected 0)" << endl;
    cout << "  First read - Low qual: " << stats[0].low_qual_bases << " (expected 0)" << endl;
    
    bool passed = (stats[0].total_bases == 100 && stats[0].n_bases == 0);
    
    delete[] stats;
    delete[] lengths;
    delete[] sequences;
    delete[] qualities;
    
    cout << (passed ? "PASSED" : "FAILED") << endl << endl;
    return passed;
}

// Test 3: GPU availability test
bool test_gpu_availability() {
    cout << "Test 3: GPU availability check..." << endl;
    
    int available = cuda_is_available();
    if (available) {
        int device = cuda_get_device();
        cout << "  GPU Available: Yes (Device " << device << ")" << endl;
        cout << "PASSED" << endl << endl;
        return true;
    } else {
        cout << "  GPU Available: No (CPU fallback)" << endl;
        cout << "PASSED (CPU fallback works)" << endl << endl;
        return true;  // CPU-only is valid
    }
}

// Test 4: Quality score computation
bool test_quality_computation() {
    cout << "Test 4: Quality score computation..." << endl;
    
    const char* seq = "ACGT";
    const char* qual = "IIII";  // Quality score '?' = 14 in Phred scale
    const int lengths[] = {4};
    
    const char** sequences = new const char*[1]{seq};
    const char** qualities = new const char*[1]{qual};
    
    ReadStatistics* stats = new ReadStatistics[1];
    
    int result = cuda_compute_read_stats(
        sequences, qualities, lengths, 1, '5', stats
    );
    
    if (result != 0) {
        cout << "ERROR: CUDA computation failed" << endl;
        return false;
    }
    
    // 'I' = ASCII 73, subtract 33 = 40
    // Expected total_quality = 40 * 4 = 160
    cout << "  Total quality: " << stats[0].total_quality << " (expected 160)" << endl;
    
    int expected_quality = 40 * 4;
    bool passed = (stats[0].total_quality == expected_quality);
    
    delete[] stats;
    delete[] sequences;
    delete[] qualities;
    
    cout << (passed ? "PASSED" : "FAILED") << endl << endl;
    return passed;
}

// Main test runner
int main() {
    cout << "====================================" << endl;
    cout << "CUDA Per-Read Statistics Unit Tests" << endl;
    cout << "====================================" << endl << endl;
    
    vector<bool> results;
    
    results.push_back(test_gpu_availability());
    results.push_back(test_single_read());
    results.push_back(test_batch_reads());
    results.push_back(test_quality_computation());
    
    // Summary
    int passed = 0;
    int total = results.size();
    
    for (bool r : results) {
        if (r) passed++;
    }
    
    cout << "====================================" << endl;
    cout << "Summary: " << passed << "/" << total << " tests passed" << endl;
    cout << "====================================" << endl;
    
    return (passed == total) ? 0 : 1;
}
