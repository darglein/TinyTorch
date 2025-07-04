function(GroupSourcesByFolder target)
    set(SOURCE_GROUP_DELIMITER "/")
    set(last_dir "")
    set(files "")

    get_target_property(sources ${target} SOURCES)

    foreach (file ${sources})
        file(RELATIVE_PATH relative_file "${PROJECT_SOURCE_DIR}" ${file})
        get_filename_component(dir "${relative_file}" PATH)
        if (NOT "${dir}" STREQUAL "${last_dir}")
            if (files)
                source_group("${last_dir}" FILES ${files})
            endif ()
            set(files "")
        endif ()
        set(files ${files} ${file})
        set(last_dir "${dir}")
    endforeach ()

    if (files)
        source_group("${last_dir}" FILES ${files})
    endif ()
endfunction()


#############  source #############

FILE(GLOB_RECURSE ALL_SRC *.cpp *.cu)
FILE(GLOB_RECURSE ALL_HEADER *.h)


add_library(torch SHARED ${ALL_HEADER} ${ALL_SRC})

if (MSVC)
    GroupSourcesByFolder(torch)
endif ()

target_include_directories(torch PUBLIC ".")

target_compile_options(torch PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=186>) # pointless comparison of unsigned integer with zero
target_compile_options(torch PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=1394>) # field of class type without a DLL interface used in a class with a DLL interface


if(OPENMP_FOUND)
    target_include_directories(torch PRIVATE ${OPENMP_INCLUDE_DIRS})
    target_compile_options(torch PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>)
    target_link_libraries(torch PRIVATE "${OpenMP_CXX_FLAGS}")
endif()

if (UNIX)
    #target_compile_options(torch PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-absolute-value>)
    #target_compile_options(torch PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wno-absolute-value>)
    target_compile_options(torch PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fvisibility=hidden>)
    target_compile_options(torch PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fvisibility=hidden>)
    target_compile_options(torch PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>) # for profiling
    target_compile_options(torch PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
endif ()

if (UNIX OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(torch PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-msse4.1>)
    target_compile_options(torch PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx>)
    target_compile_options(torch PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2>)
    target_compile_options(torch PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mfma>)
endif()


if (MSVC)
#    target_compile_options(torch PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/fp:fast>)
#    target_compile_options(torch PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/fp:fast>)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # make sure libopenmp is used also for the .cu files instead of the msvc openmp
    target_compile_options(torch PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/openmp:llvm>)
    target_compile_options(torch PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/openmp:llvm>)
else()
    target_compile_options(torch PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/openmp>)
    target_compile_options(torch PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/openmp>)
endif ()

    target_link_libraries(torch PRIVATE OpenMP::OpenMP_CXX)


    # '...' needs to have dll-interface to be used by clients of struct '...'
    target_compile_options(torch PUBLIC $<$<COMPILE_LANGUAGE:CXX>:/wd4251>)
    target_compile_options(torch PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4251>)
    
    # unknown pragma 'unroll'
    target_compile_options(torch PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/wd4068>)
    target_compile_options(torch PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4068>)
#    target_compile_options(torch PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/openmp>)

    target_link_libraries(torch PRIVATE DbgHelp)
endif ()




target_link_libraries(torch PUBLIC glog::glog)
#target_link_libraries(torch PRIVATE "${OpenMP_CXX_LIB_NAMES}")



if (TT_HAS_CUDA)
    target_link_libraries(torch PUBLIC CUDA::cudart)
    set_target_properties(torch PROPERTIES CUDA_ARCHITECTURES "75;75-virtual;89;89-virtual")

    target_compile_options(torch PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endif ()


set(MY_BUILDCONFIG_DIR "${CMAKE_CURRENT_BINARY_DIR}/include")
configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/torch/tiny_torch_build_config.h.in"
        "${MY_BUILDCONFIG_DIR}/torch/tiny_torch_build_config.h"
        @ONLY
)
target_include_directories(torch PUBLIC ${MY_BUILDCONFIG_DIR})


if (MSVC)
    target_sources(torch PRIVATE torch.natvis)
endif ()


#############  output dir ############# 

#set_target_properties( ${APP_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set_property(TARGET torch PROPERTY CXX_STANDARD 17)


