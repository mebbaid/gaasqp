# Benchmarks comparing GeoQP with other QP solvers

find_package(QpSolversEigen QUIET)
find_package(qpOASES QUIET)

# RANDOM GENERATED PROBLEMS AND LIPM-MPC-LIKE Problem
if(qpOASES_FOUND)
    message(STATUS "Building box-focused benchmark")

    add_executable(benchmark_box benchmark-box.cpp)
    target_include_directories(benchmark_box PRIVATE ${qpOASES_INCLUDE_DIR})
    target_link_libraries(benchmark_box PRIVATE
        geoqp
        ${qpOASES_LIBRARY})
    target_compile_features(benchmark_box PRIVATE cxx_std_17)

    install(TARGETS benchmark_box
        RUNTIME DESTINATION bin)
endif()

if(QpSolversEigen_FOUND AND qpOASES_FOUND)
    message(STATUS "Building Unified Benchmark")

    add_executable(benchmark benchmark.cpp)
    target_include_directories(benchmark PRIVATE ${qpOASES_INCLUDE_DIR})
    target_link_libraries(benchmark PRIVATE 
        geoqp 
        QpSolversEigen::QpSolversEigen
        ${qpOASES_LIBRARY})
    target_compile_features(benchmark PRIVATE cxx_std_17)
    
    install(TARGETS benchmark
        RUNTIME DESTINATION bin)

    message(STATUS "Building LIPM walking controller benchmark")
    
    add_executable(benchmark_lipm_walking benchmark_lipm_walking.cpp)
    target_include_directories(benchmark_lipm_walking PRIVATE ${qpOASES_INCLUDE_DIR})
    target_link_libraries(benchmark_lipm_walking PRIVATE 
        geoqp 
        QpSolversEigen::QpSolversEigen
        ${qpOASES_LIBRARY})
    target_compile_features(benchmark_lipm_walking PRIVATE cxx_std_17)
    
    install(TARGETS benchmark_lipm_walking
        RUNTIME DESTINATION bin)
else()
    message(STATUS "Skipping LIPM walking benchmark (requires both OSQP and qpOASES)")
endif()
