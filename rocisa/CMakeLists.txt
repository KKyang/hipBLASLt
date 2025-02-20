cmake_minimum_required(VERSION 3.15)
project(rocisa LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_C_COMPILER c)
set(CMAKE_CXX_COMPILER c++)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# find nanobind
execute_process(
    COMMAND python3 -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_DIR
    ERROR_VARIABLE error
    RESULT_VARIABLE result
)
message(STATUS "Nanobind path: ${nanobind_DIR}")
string(REPLACE "\n" "" nanobind_DIR "${nanobind_DIR}")

if(DEFINED PYTHON_VERSION AND PYTHON_VERSION)
    message(STATUS "Manually set Python version to ${PYTHON_VERSION}")
else()
    set(PYTHON_VERSION 3)
endif()

# Use cmake -Dpybind11_DIR=$(python3 -m nanobind --cmakedir) .. to set pybind dir
# Use cmake -DCMAKE_PREFIX_PATH=$(python3 -m nanobind --cmake_dir) .. to set nanobind

find_package(Python ${PYTHON_VERSION} COMPONENTS Interpreter Development REQUIRED)
find_package(nanobind CONFIG REQUIRED HINTS ${nanobind_DIR})

include_directories(${CMAKE_SOURCE_DIR}/include)
nanobind_add_module(rocisa src/base.cpp)
