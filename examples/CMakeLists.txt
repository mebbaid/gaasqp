add_executable(simple_example simple_example.cpp)
target_link_libraries(simple_example PRIVATE geoqp::geoqp)

add_executable(mpc_example mpc_example.cpp)
target_link_libraries(mpc_example PRIVATE geoqp::geoqp)

add_executable(timing_test timing_test.cpp)
target_link_libraries(timing_test PRIVATE geoqp::geoqp)

# Planar comparison example (requires both OSQP and qpOASES)
find_package(QpSolversEigen QUIET)
find_package(qpOASES QUIET)

if(QpSolversEigen_FOUND AND qpOASES_FOUND)
    add_executable(planar_comparison planar_comparison.cpp)
    target_include_directories(planar_comparison PRIVATE ${qpOASES_INCLUDE_DIR})
    target_link_libraries(planar_comparison PRIVATE 
        geoqp::geoqp 
        QpSolversEigen::QpSolversEigen
        ${qpOASES_LIBRARY}
    )
    target_compile_features(planar_comparison PRIVATE cxx_std_17)
    
    install(TARGETS planar_comparison
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/examples
    )
endif()

# Install examples
install(TARGETS simple_example mpc_example timing_test
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/examples
)
