cmake_minimum_required(VERSION 2.8)
project(NaiveBayes360)

ADD_DEFINITIONS(
    -std=c++11 # Or -std=c++0x
    # Other flags
)
find_package( Threads )


set(SOURCE_FILES
        main.cpp
        bitmap.cpp
        bitmap.hpp
        mnist_reader_common.hpp
        mnist_reader_less.hpp
        mnist_utils.hpp
        mnist_reader.hpp)

add_executable(NaiveBayes360 ${SOURCE_FILES})
target_link_libraries(NaiveBayes360 ${CMAKE_THREAD_LIBS_INIT} )
