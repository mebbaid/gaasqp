# Google Test setup
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1
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

# Register tests
include(GoogleTest)
gtest_discover_tests(test_linear_algebra)
gtest_discover_tests(test_utilities)
gtest_discover_tests(test_integration)
gtest_discover_tests(test_schur_complement)
