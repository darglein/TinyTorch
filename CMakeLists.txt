cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(TinyTorch VERSION 1.0.0 LANGUAGES CXX)


############# C++ Standard and Filesystem stuff ###############

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(TT_BUILD_SAMPLES "Build the tt samples" ON)

find_package(CUDAToolkit 11.8)
if (CUDAToolkit_FOUND)
    enable_language(CUDA)
    set(TT_HAS_CUDA 1)
    message(STATUS "Enable CUDA support")
else ()
    set(TT_HAS_CUDA 0)
    message(STATUS "Disable CUDA support (CUDAToolkit not found)")
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







