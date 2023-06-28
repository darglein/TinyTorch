cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project (TinyTorch VERSION 1.0.0 LANGUAGES CXX)


############# C++ Standard and Filesystem stuff ###############

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



############# SOURCE ###############

add_subdirectory(src)
add_subdirectory(samples)







