

add_library(torch SHARED "")
target_include_directories(torch PUBLIC ".")

if (UNIX)
    target_compile_options(torch PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fvisibility=hidden>)
    target_compile_options(torch PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fvisibility=hidden>)
endif ()


target_link_libraries(torch PUBLIC glog::glog)

if (TT_HAS_CUDA)
    target_link_libraries(torch PUBLIC CUDA::cudart)
    set_target_properties(torch PROPERTIES CUDA_ARCHITECTURES "72;72-virtual")
endif ()


set(MY_BUILDCONFIG_DIR "${CMAKE_CURRENT_BINARY_DIR}/include")
configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/torch/tiny_torch_build_config.h.in"
        "${MY_BUILDCONFIG_DIR}/torch/tiny_torch_build_config.h"
        @ONLY
)
target_include_directories(torch PUBLIC ${MY_BUILDCONFIG_DIR})


#############  source #############

FILE(GLOB_RECURSE ALL_SRC *.cpp *.cu)
FILE(GLOB_RECURSE ALL_HEADER *.h)

target_sources(torch PRIVATE ${ALL_SRC} PUBLIC ${ALL_HEADER})


#############  output dir ############# 

#set_target_properties( ${APP_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set_property(TARGET torch PROPERTY CXX_STANDARD 17)


