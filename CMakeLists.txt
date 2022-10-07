cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

set(LIB_NAME TinyTorch)
project (${LIB_NAME} VERSION 1.0.0 LANGUAGES CXX)


set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH} )






############# Required LIBRARIES ###############



############# COMPILER FLAGS ###############



############# C++ Standard and Filesystem stuff ###############

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Gcc needs this extra lib for linking
if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 8.0)
    SET(LIBS ${LIBS} stdc++fs)
endif()



############# SOURCE ###############

add_subdirectory(src)







