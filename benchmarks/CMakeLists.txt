# Benchmarks comparing GeoQP with other QP solvers

find_package(QpSolversEigen QUIET)
find_package(qpOASES QUIET)

# Face hopping strategy comparison (always build - no dependencies)
add_executable(compare_face_hopping compare_face_hopping.cpp)
target_link_libraries(compare_face_hopping PRIVATE geoqp)
target_compile_features(compare_face_hopping PRIVATE cxx_std_17)

install(TARGETS compare_face_hopping
    RUNTIME DESTINATION bin)



# RANDOM GENERATED PROBLEMS AND LIPM-MPC
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
