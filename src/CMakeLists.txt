function(GroupSourcesByFolder target)
  set(SOURCE_GROUP_DELIMITER "/")
  set(last_dir "")
  set(files "")

  get_target_property(sources ${target} SOURCES)

  foreach(file ${sources})
    file(RELATIVE_PATH relative_file "${PROJECT_SOURCE_DIR}" ${file})
    get_filename_component(dir "${relative_file}" PATH)
    if(NOT "${dir}" STREQUAL "${last_dir}")
      if(files)
        source_group("${last_dir}" FILES ${files})
      endif()
      set(files "")
    endif()
    set(files ${files} ${file})
    set(last_dir "${dir}")
  endforeach()

  if(files)
    source_group("${last_dir}" FILES ${files})
  endif()
endfunction()



#############  source #############

FILE(GLOB_RECURSE ALL_SRC *.cpp *.cu)
FILE(GLOB_RECURSE ALL_HEADER *.h)


add_library(torch SHARED ${ALL_HEADER} ${ALL_SRC})

if(MSVC)
    GroupSourcesByFolder(torch)
endif()

target_include_directories(torch PUBLIC ".")

if (UNIX)
    target_compile_options(torch PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fvisibility=hidden>)
    target_compile_options(torch PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fvisibility=hidden>)
endif ()


target_link_libraries(torch PUBLIC glog::glog)

if (TT_HAS_CUDA)
    target_link_libraries(torch PUBLIC CUDA::cudart)
    set_target_properties(torch PROPERTIES CUDA_ARCHITECTURES "72;72-virtual")

    target_compile_options(torch PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endif ()


set(MY_BUILDCONFIG_DIR "${CMAKE_CURRENT_BINARY_DIR}/include")
configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/torch/tiny_torch_build_config.h.in"
        "${MY_BUILDCONFIG_DIR}/torch/tiny_torch_build_config.h"
        @ONLY
)
target_include_directories(torch PUBLIC ${MY_BUILDCONFIG_DIR})


if(MSVC)
    target_sources(torch PRIVATE torch.natvis)
endif()


#############  output dir ############# 

#set_target_properties( ${APP_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set_property(TARGET torch PROPERTY CXX_STANDARD 17)


