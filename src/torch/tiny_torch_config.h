/**
* Copyright (c) 2022 Darius Rückert
* Licensed under the MIT License.
* See LICENSE file for more information.
 */

#pragma once

#include "glog/logging.h"
#include "torch/tiny_torch_build_config.h"

// Generic helper definitions for shared library support
#if defined _WIN32 || defined __CYGWIN__
#    define TINYTORCH_HELPER_DLL_IMPORT __declspec(dllimport)
#    define TINYTORCH_HELPER_DLL_EXPORT __declspec(dllexport)
#    define TINYTORCH_HELPER_DLL_LOCAL
#else
#    if __GNUC__ >= 4  // Note: Clang also defines GNUC
#        define TINYTORCH_HELPER_DLL_IMPORT __attribute__((visibility("default")))
#        define TINYTORCH_HELPER_DLL_EXPORT __attribute__((visibility("default")))
#        define TINYTORCH_HELPER_DLL_LOCAL __attribute__((visibility("hidden")))
#    else
#        error Unknown import/export defines.
#        define TINYTORCH_HELPER_DLL_IMPORT
#        define TINYTORCH_HELPER_DLL_EXPORT
#        define TINYTORCH_HELPER_DLL_LOCAL
#    endif
#endif

#ifdef torch_EXPORTS
#    define TINYTORCH_API TINYTORCH_HELPER_DLL_EXPORT
#else
#    define TINYTORCH_API TINYTORCH_HELPER_DLL_IMPORT
#endif

#ifndef TINY_TORCH_NAMESPACE
#define TINY_TORCH_NAMESPACE tinytorch
#endif

#ifdef TT_HAS_CUDA
#    include <cuda_runtime_api.h>
#    define TT_HD __host__ __device__
#else
#    define TT_HD
#endif



#if defined _WIN32
#    define TT_FUNCTION_NAME __FUNCSIG__
#elif defined __unix__
#    include <features.h>
#    if defined __cplusplus ? __GNUC_PREREQ(2, 6) : __GNUC_PREREQ(2, 4)
#        define TT_FUNCTION_NAME __PRETTY_FUNCTION__
#    else
#        if defined __STDC_VERSION__ && __STDC_VERSION__ >= 199901L
#            define TT_FUNCTION_NAME __func__
#        else
#            define TT_FUNCTION_NAME ((const char*)0)
#        endif
#    endif
#elif defined __APPLE__
#    define TT_FUNCTION_NAME __PRETTY_FUNCTION__
#else
#    error Unknown compiler.
#endif


#ifdef NDEBUG
#define TT_DEBUG 0
#else
#define TT_DEBUG 1
#endif