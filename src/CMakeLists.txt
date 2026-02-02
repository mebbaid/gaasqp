set(GEOQP_SOURCES
    solver.cpp
    linear_algebra.cpp
    polytope.cpp
    initializer.cpp
    face_hopper.cpp
    projection_solver.cpp
    projector.cpp
    kkt_checker.cpp
    utilities.cpp
    schur_complement.cpp
    c_api.cpp
)

add_library(geoqp ${GEOQP_SOURCES})
add_library(geoqp::geoqp ALIAS geoqp)

# Enable position-independent code for static library
set_target_properties(geoqp PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(geoqp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

if(GEOQP_USE_EIGEN)
    target_link_libraries(geoqp PUBLIC Eigen3::Eigen)
    target_compile_definitions(geoqp PUBLIC GEOQP_USE_EIGEN)
endif()

if(GEOQP_USE_BLAS)
    target_link_libraries(geoqp PUBLIC ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
    target_compile_definitions(geoqp PUBLIC GEOQP_USE_BLAS)
endif()

# C lib
add_library(geoqp_c SHARED c_api.cpp)
target_link_libraries(geoqp_c PUBLIC geoqp)
target_include_directories(geoqp_c
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

install(TARGETS geoqp geoqp_c
    EXPORT geoqp-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(EXPORT geoqp-targets
    FILE geoqp-targets.cmake
    NAMESPACE geoqp::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/geoqp
)
