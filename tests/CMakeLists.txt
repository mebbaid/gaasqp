# Google Test setup
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# Test files
add_executable(test_linear_algebra test_linear_algebra.cpp)
target_link_libraries(test_linear_algebra PRIVATE geoqp::geoqp GTest::gtest_main)

add_executable(test_utilities test_utilities.cpp)
target_link_libraries(test_utilities PRIVATE geoqp::geoqp GTest::gtest_main)

add_executable(test_integration test_integration.cpp)
target_link_libraries(test_integration PRIVATE geoqp::geoqp GTest::gtest_main)

add_executable(test_schur_complement test_schur_complement.cpp)
target_link_libraries(test_schur_complement PRIVATE geoqp::geoqp GTest::gtest_main)

# Visible-facet geometry (uses Eigen for an independent reference QP solver)
if(GEOQP_USE_EIGEN)
    add_executable(test_box_geometry test_box_geometry.cpp)
    target_link_libraries(test_box_geometry PRIVATE geoqp::geoqp GTest::gtest_main)
endif()

# Register tests
include(GoogleTest)
if(GEOQP_USE_EIGEN)
    gtest_discover_tests(test_box_geometry)
endif()
gtest_discover_tests(test_linear_algebra)
gtest_discover_tests(test_utilities)
gtest_discover_tests(test_integration)
gtest_discover_tests(test_schur_complement)
