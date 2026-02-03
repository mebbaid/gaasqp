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

# Find qpOASES for active set comparison
find_package(qpOASES QUIET)

if(qpOASES_FOUND)
    message(STATUS "qpOASES found - building active set benchmark")
    
    add_executable(benchmark_qpoases benchmark_qpoases.cpp)
    target_include_directories(benchmark_qpoases PRIVATE ${qpOASES_INCLUDE_DIR})
    target_link_libraries(benchmark_qpoases PRIVATE geoqp ${qpOASES_LIBRARY})
    target_compile_features(benchmark_qpoases PRIVATE cxx_std_17)
    
    install(TARGETS benchmark_qpoases
        RUNTIME DESTINATION bin)
else()
    message(STATUS "qpOASES not found - skipping active set benchmark")
    message(STATUS "To enable: pixi add qpoases")
endif()

# Face hopping strategy comparison (always build - no dependencies)
add_executable(compare_face_hopping compare_face_hopping.cpp)
target_link_libraries(compare_face_hopping PRIVATE geoqp)
target_compile_features(compare_face_hopping PRIVATE cxx_std_17)

install(TARGETS compare_face_hopping
    RUNTIME DESTINATION bin)
