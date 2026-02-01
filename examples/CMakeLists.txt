add_executable(simple_example simple_example.cpp)
target_link_libraries(simple_example PRIVATE geoqp::geoqp)

add_executable(mpc_example mpc_example.cpp)
target_link_libraries(mpc_example PRIVATE geoqp::geoqp)

# Install examples
install(TARGETS simple_example mpc_example
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/examples
)
