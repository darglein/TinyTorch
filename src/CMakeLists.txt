
set(APP_NAME tt_sample)


add_executable(${APP_NAME} "")
target_include_directories(${APP_NAME} PUBLIC ".")


#############  source #############

FILE(GLOB_RECURSE ALL_SRC  *.cpp)
FILE(GLOB_RECURSE ALL_HEADER  *.h)

target_sources(${APP_NAME} PRIVATE ${ALL_SRC} PUBLIC ${ALL_HEADER})


#############  output dir ############# 

set_target_properties( ${APP_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set_property(TARGET ${APP_NAME} PROPERTY CXX_STANDARD 17)


