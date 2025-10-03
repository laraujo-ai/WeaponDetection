cmake_minimum_required(VERSION 3.10)
project(WeaponDetectionAPI VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_library(ONNXRUNTIME_LIB onnxruntime HINTS /usr/local/lib)
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h HINTS /usr/local/include/onnxruntime)

if(NOT ONNXRUNTIME_LIB OR NOT ONNXRUNTIME_INCLUDE_DIR)
    message(FATAL_ERROR "ONNX Runtime not found. Please run install_onnxruntime.sh first.")
endif()

add_executable(${PROJECT_NAME} src/main.cpp)

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION ${ONNXRUNTIME_LIB}
    INTERFACE_INCLUDE_DIRECTORIES ${ONNXRUNTIME_INCLUDE_DIR}
)

find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
else()
    message(STATUS "OpenCV not found - image handling will be limited")
endif()

add_subdirectory(src)
target_link_libraries(${PROJECT_NAME} PRIVATE onnxruntime ${OpenCV_LIBS} ObjectDetection common)