cmake_minimum_required(VERSION 3.10)
project(ObjectDetection)

add_library(${PROJECT_NAME} STATIC
    include/YOLOXDetector.h
    src/YOLOXDetector.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC include)
target_link_libraries(${PROJECT_NAME} PUBLIC common ${OpenCV_LIBS})
