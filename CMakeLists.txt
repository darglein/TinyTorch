cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(TinyTorch VERSION 1.0.0 LANGUAGES CXX)


############# C++ Standard and Filesystem stuff ###############

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(TT_BUILD_SAMPLES "Build the tt samples" ON)
option(TT_WITH_CUDA "Enable CUDA if possible" ON)
option(TT_ALL_OUTPUT_TO_BIN "Writes all .dll and .exe to the bin directory" OFF)

if (TT_WITH_CUDA)
    find_package(CUDAToolkit 11.8)
endif ()

if (CUDAToolkit_FOUND)
    enable_language(CUDA)
    set(TT_HAS_CUDA 1)
    message(STATUS "Enable CUDA support")
else ()
    set(TT_HAS_CUDA 0)
    message(STATUS "Disable CUDA support (CUDAToolkit not found)")
endif ()

if (TT_ALL_OUTPUT_TO_BIN)
    SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
endif ()


find_package(OpenMP REQUIRED)

#### Submodules ####

if (NOT TARGET glog::glog)
    set(WITH_GFLAGS OFF CACHE INTERNAL "")
    set(WITH_GTEST OFF CACHE INTERNAL "")
    set(WITH_UNWIND OFF CACHE INTERNAL "")
    add_subdirectory(External/glog)
endif ()

############# SOURCE ###############

add_subdirectory(src)

if (TT_BUILD_SAMPLES)
    add_subdirectory(samples)
endif ()







