# Benchmarks comparing GeoQP with other QP solvers

# Find QpSolversEigen (provides OSQP interface)
find_package(QpSolversEigen QUIET)

if(QpSolversEigen_FOUND)
    message(STATUS "QpSolversEigen found - building OSQP benchmark")
    
    add_executable(benchmark_osqp benchmark_osqp.cpp)
    target_link_libraries(benchmark_osqp PRIVATE geoqp QpSolversEigen::QpSolversEigen)
    target_compile_features(benchmark_osqp PRIVATE cxx_std_17)
    
    # Install benchmark
    install(TARGETS benchmark_osqp
        RUNTIME DESTINATION bin)
else()
    message(STATUS "QpSolversEigen not found - skipping OSQP benchmark")
    message(STATUS "To enable: pixi add qpsolvers-eigen")
endif()
