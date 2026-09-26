cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(TinyTorch VERSION 1.0.0 LANGUAGES CXX)


############# C++ Standard and Filesystem stuff ###############

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(TT_BUILD_SAMPLES "Build the tt samples" ON)
option(TT_BUILD_TESTS "Build the tt tests (requires the googletest submodule)" ON)
option(TT_WITH_CUDA "Enable CUDA if possible" ON)
option(TT_ALL_OUTPUT_TO_BIN "Writes all .dll and .exe to the bin directory" OFF)

if (TT_WITH_CUDA)
    find_package(CUDAToolkit 11.8)
endif ()


if (CUDAToolkit_FOUND)
    enable_language(CUDA)
    set(TT_HAS_CUDA 1)
    message(STATUS "Enable CUDA support")

    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
    include(select_compute_arch)

    if (TT_CUDA_ARCH)
        set(TT_CUDA_ARCH_LIST "${TT_CUDA_ARCH}")
        message(STATUS "TinyTorch CUDA arch (from parent): " "${TT_CUDA_ARCH}")
    else ()
        message(STATUS "TinyTorch CUDA arch autodetection...")
        CUDA_DETECT_INSTALLED_GPUS(TT_CUDA_ARCH_LIST)
    endif ()

    # Convert to the CMake CUDA_ARCHITECTURES notation: real (sm_N) plus virtual
    # (PTX) code for every arch, so the library also runs on slightly newer GPUs.
    set(TT_CUDA_ARCHITECTURES "")
    foreach (arch ${TT_CUDA_ARCH_LIST})
        if (arch MATCHES "^([0-9][0-9][0-9]?)([af])?(\\+PTX)?$")
            set(arch_base "${CMAKE_MATCH_1}${CMAKE_MATCH_2}")
            list(APPEND TT_CUDA_ARCHITECTURES ${arch_base} ${arch_base}-virtual)
        else ()
            message(WARNING "Ignoring invalid CUDA arch token '${arch}' (expected e.g. 75, 90a, 100f, 89+PTX)")
        endif ()
    endforeach ()
    list(REMOVE_DUPLICATES TT_CUDA_ARCHITECTURES)
    message(STATUS "TinyTorch CUDA_ARCHITECTURES: " "${TT_CUDA_ARCHITECTURES}")
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
    add_subdirectory(External/tiny-glog)
endif ()

############# SOURCE ###############

add_subdirectory(src)

if (TT_BUILD_SAMPLES)
    add_subdirectory(samples)
endif ()

############# TESTS ###############

if (TT_BUILD_TESTS)
    include(CTest)

    if (NOT TARGET gtest)
        set(BUILD_GTEST ON CACHE BOOL "" FORCE)
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        add_subdirectory(External/googletest EXCLUDE_FROM_ALL)
    endif ()

    add_subdirectory(tests)
endif ()







