cmake_minimum_required(VERSION 3.6)
project(NaiveBayes360)

set(CMAKE_CXX_STANDARD 11)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
set(SOURCE_FILES
        main.cpp
        bitmap.cpp
        bitmap.hpp
        mnist_reader_common.hpp
        mnist_reader_less.hpp
        mnist_utils.hpp
        mnist_reader.hpp)

add_executable(NaiveBayes360 ${SOURCE_FILES})

