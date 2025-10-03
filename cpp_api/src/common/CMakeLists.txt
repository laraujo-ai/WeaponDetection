cmake_minimum_required(VERSION 3.10)
project(common)

add_library(${PROJECT_NAME} STATIC
    include/BaseModel.hpp
    include/SessionBuilder.hpp
    src/SessionBuilder.cpp
    src/BaseModel.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC include)
target_link_libraries(${PROJECT_NAME} PUBLIC onnxruntime)
