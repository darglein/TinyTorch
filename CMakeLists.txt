cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(TinyTorch VERSION 1.0.0 LANGUAGES CXX)


############# C++ Standard and Filesystem stuff ###############

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(CUDAToolkit 11.8)
if (CUDAToolkit_FOUND)
    enable_language(CUDA)
    set(TT_HAS_CUDA 1)
    message(STATUS "Enable CUDA support")
else ()
    set(TT_HAS_CUDA 0)
    message(STATUS "Disable CUDA support (CUDAToolkit not found)")
endif ()


#### Submodules ####

set(WITH_GFLAGS OFF CACHE INTERNAL "")
set(WITH_GTEST OFF CACHE INTERNAL "")
set(WITH_UNWIND OFF CACHE INTERNAL "")
add_subdirectory(External/glog)

############# SOURCE ###############

add_subdirectory(src)
add_subdirectory(samples)







