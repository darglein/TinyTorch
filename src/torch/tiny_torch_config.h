/**
* Copyright (c) 2022 Darius Rückert
* Licensed under the MIT License.
* See LICENSE file for more information.
 */

#pragma once

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