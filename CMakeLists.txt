cmake_minimum_required(VERSION 3.10)
project(fastp CUDA CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -g -O3")

# Auto-increment patch version on source code changes
message(STATUS "Checking version...")
execute_process(
    COMMAND bash ${PROJECT_SOURCE_DIR}/scripts/version_manager.sh check
    OUTPUT_VARIABLE VERSION_CHECK_OUTPUT
    RESULT_VARIABLE VERSION_CHECK_RESULT
)
message(STATUS "${VERSION_CHECK_OUTPUT}")

# If source files changed, auto-increment patch version
if(VERSION_CHECK_OUTPUT MATCHES "changed")
    message(STATUS "Incrementing patch version automatically...")
    execute_process(
        COMMAND bash ${PROJECT_SOURCE_DIR}/scripts/version_manager.sh increment
        OUTPUT_VARIABLE INCREMENT_OUTPUT
    )
    message(STATUS "${INCREMENT_OUTPUT}")
endif()

# Find CUDA
find_package(CUDA QUIET)

# Source files
set(DIR_SRC src)
set(DIR_OBJ obj)

file(GLOB CPP_SOURCES "${DIR_SRC}/*.cpp")
file(GLOB CU_SOURCES "${DIR_SRC}/*.cu")

# Exclude test file from main build (will be built separately)
list(FILTER CU_SOURCES EXCLUDE REGEX "test_cuda_stats")

# If CUDA is found, enable GPU acceleration
if(CUDA_FOUND)
    message(STATUS "CUDA found: Version ${CUDA_VERSION}")
    message(STATUS "Building fastp with CUDA GPU acceleration for per-read statistics")
    
    set(CUDA_SEPARABLE_COMPILATION ON)
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    
    # CUDA compiler flags
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
    
    # Detect GPU compute capability
    if(NOT CUDA_ARCH)
        set(CUDA_ARCH "61")  # Default to compute capability 6.1 (GTX 1050/GTX 1080 era)
    endif()
    
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_${CUDA_ARCH},code=compute_${CUDA_ARCH}")
    
    # Add CUDA sources
    list(APPEND SOURCES ${CPP_SOURCES})
    list(APPEND SOURCES ${CU_SOURCES})
    
    set(CUDA_ENABLED TRUE)
else()
    message(STATUS "CUDA not found. Building fastp without GPU acceleration")
    message(STATUS "To enable GPU acceleration, install CUDA Toolkit and rerun CMake")
    
    # Only use C++ sources
    list(APPEND SOURCES ${CPP_SOURCES})
    set(CUDA_ENABLED FALSE)
endif()

# Find required libraries
find_library(ISAL_LIB isal)
find_library(DEFLATE_LIB deflate)
find_package(Threads REQUIRED)

# Create executable
add_executable(fastp ${SOURCES})

# Include directories
target_include_directories(fastp PRIVATE ${DIR_SRC})
target_include_directories(fastp PRIVATE ${DIR_SRC}/libs)

# Link libraries
target_link_libraries(fastp PRIVATE ${ISAL_LIB} ${DEFLATE_LIB} Threads::Threads)

if(CUDA_FOUND)
    target_link_libraries(fastp PRIVATE ${CUDA_LIBRARIES})
    target_include_directories(fastp PRIVATE ${CUDA_INCLUDE_DIRS})
    target_compile_definitions(fastp PRIVATE CUDA_ENABLED)
endif()

# Installation
install(TARGETS fastp DESTINATION bin)

# Print summary
message(STATUS "")
message(STATUS "fastp Configuration Summary:")
message(STATUS "  CUDA Support: ${CUDA_ENABLED}")
if(CUDA_FOUND)
    message(STATUS "  CUDA Version: ${CUDA_VERSION}")
    message(STATUS "  CUDA Toolkit Path: ${CUDA_TOOLKIT_ROOT_DIR}")
    message(STATUS "  GPU Compute Capability: ${CUDA_ARCH}")
endif()
message(STATUS "")
