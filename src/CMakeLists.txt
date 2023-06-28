


add_library(torch SHARED "")
target_include_directories(torch PUBLIC ".")

if (UNIX)
    target_compile_options(torch PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fvisibility=hidden>)
    target_compile_options(torch PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fvisibility=hidden>)
endif ()

#############  source #############

FILE(GLOB_RECURSE ALL_SRC  *.cpp)
FILE(GLOB_RECURSE ALL_HEADER  *.h)

target_sources(torch PRIVATE ${ALL_SRC} PUBLIC ${ALL_HEADER})


#############  output dir ############# 

#set_target_properties( ${APP_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set_property(TARGET torch PROPERTY CXX_STANDARD 17)


