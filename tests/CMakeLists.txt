set(APP_NAME test_tiny_torch_all)

add_executable(${APP_NAME} "")
target_include_directories(${APP_NAME} PUBLIC ".")
target_link_libraries(${APP_NAME} PUBLIC torch gtest_main)
# A parent project may build gtest as a DLL, but gtest only adds GTEST_LINKED_AS_SHARED_LIBRARY to its install
# interface. Without it, gtest's flag variables (GTEST_FLAG_SET) are not dllimport-ed and fail to link on Windows.
get_target_property(GTEST_TYPE gtest TYPE)
if (GTEST_TYPE STREQUAL "SHARED_LIBRARY")
    target_compile_definitions(${APP_NAME} PRIVATE GTEST_LINKED_AS_SHARED_LIBRARY=1)
endif ()

#############  source #############

FILE(GLOB_RECURSE ALL_SRC  *.cpp)
FILE(GLOB_RECURSE ALL_HEADER  *.h)

target_sources(${APP_NAME} PRIVATE ${ALL_SRC} PUBLIC ${ALL_HEADER})

#############  output dir #############

set_target_properties( ${APP_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set_property(TARGET ${APP_NAME} PROPERTY CXX_STANDARD 17)

#############  ctest #############

#add_test(NAME ${APP_NAME} COMMAND ${APP_NAME})

include(GoogleTest)
gtest_discover_tests(${APP_NAME} DISCOVERY_MODE PRE_TEST)