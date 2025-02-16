cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(BlobDetection)

set(CMAKE_CXX_STANDARD 20)
set(OpenCV_DIR ~/opencv_install/lib/cmake/opencv4)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} ${test_images} )

add_executable(BlobDetection main.cpp)
target_link_libraries( BlobDetection ${OpenCV_LIBS} )