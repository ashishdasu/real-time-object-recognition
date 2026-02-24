cmake_minimum_required(VERSION 3.10)
project(ObjectRecognition)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(objrecog
    main.cpp
    threshold.cpp
    morphology.cpp
)
target_link_libraries(objrecog ${OpenCV_LIBS})
