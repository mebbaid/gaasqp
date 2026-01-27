cmake_minimum_required(VERSION 3.15)
project(geoqp VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(GEOQP_BUILD_TESTS "Build tests" ON)
option(GEOQP_BUILD_EXAMPLES "Build examples" ON)
option(GEOQP_BUILD_BENCHMARKS "Build benchmarks" ON)
option(GEOQP_BUILD_PYTHON "Build Python bindings" ON)
option(GEOQP_USE_EIGEN "Use Eigen for linear algebra" ON)
option(GEOQP_USE_BLAS "Use BLAS/LAPACK" OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -pedantic")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

if(GEOQP_USE_EIGEN)
    find_package(Eigen3 3.3 REQUIRED NO_MODULE)
endif()

if(GEOQP_USE_BLAS)
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
endif()

add_subdirectory(src)

if(GEOQP_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(GEOQP_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(GEOQP_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Python bindings
if(GEOQP_BUILD_PYTHON)
    add_subdirectory(python)
endif()

# Installation
include(GNUInstallDirs)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
