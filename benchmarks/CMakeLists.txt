find_package(QpSolversEigen QUIET)
find_package(OsqpEigen QUIET)
find_package(qpOASES QUIET)
find_package(osqp QUIET)

if(NOT qpOASES_FOUND)
    message(STATUS "Skipping benchmarks (qpOASES not found)")
    return()
endif()

# ─────────────────────────────────────────────────────────────────────────────
# Third-party solvers requested in the reviews (DAQP, PIQP, Clarabel).
# Sources are pinned to release commits; nothing is installed outside build/.
# ─────────────────────────────────────────────────────────────────────────────
option(GEOQP_BENCH_EXTRA_SOLVERS "Fetch and build DAQP, PIQP and Clarabel for the benchmarks" ON)

set(GEOQP_BENCH_HAVE_DAQP OFF)
set(GEOQP_BENCH_HAVE_PIQP OFF)
set(GEOQP_BENCH_HAVE_CLARABEL OFF)

if(GEOQP_BENCH_EXTRA_SOLVERS)
    include(FetchContent)

    # Keep PIQP's `option(BUILD_SHARED_LIBS ... ON)` from turning every later
    # library in this build (libgeoqp included, on re-configure) into a shared one.
    if(NOT DEFINED CACHE{BUILD_SHARED_LIBS})
        set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries")
    endif()

    # DAQP v0.11.0 — dual active-set QP solver (C)
    set(PROFILING OFF CACHE BOOL "DAQP: internal timing" FORCE)
    FetchContent_Declare(daqp
        GIT_REPOSITORY https://github.com/darnstrom/daqp.git
        GIT_TAG        44a429a31ebb9e9eea3a12a5d84d3eaeec8c7938 # v0.11.0
        GIT_SHALLOW    FALSE
        EXCLUDE_FROM_ALL)

    # PIQP v0.6.4 — proximal interior-point QP solver (header-only C++/Eigen)
    set(BUILD_WITH_TEMPLATE_INSTANTIATION OFF CACHE BOOL "PIQP: header-only" FORCE)
    set(BUILD_C_INTERFACE OFF CACHE BOOL "PIQP: C interface" FORCE)
    set(BUILD_TESTS OFF CACHE BOOL "PIQP: tests" FORCE)
    set(BUILD_EXAMPLES OFF CACHE BOOL "PIQP: examples" FORCE)
    set(BUILD_BENCHMARKS OFF CACHE BOOL "PIQP: benchmarks" FORCE)
    set(ENABLE_INSTALL OFF CACHE BOOL "PIQP: install targets" FORCE)
    FetchContent_Declare(piqp
        GIT_REPOSITORY https://github.com/PREDICT-EPFL/piqp.git
        GIT_TAG        5d849590a18bc7f0b449057e5f2041713a85003c # v0.6.4
        EXCLUDE_FROM_ALL)

    # Clarabel.cpp v0.11.1 — interior-point conic solver (Rust core + C++ headers).
    # Only the sources are fetched (SOURCE_SUBDIR points nowhere): its own CMake
    # runs `cargo install cbindgen` into ~/.cargo and regenerates headers that are
    # already committed.  We build the Rust static library directly instead.
    FetchContent_Declare(clarabel
        GIT_REPOSITORY https://github.com/oxfordcontrol/Clarabel.cpp.git
        GIT_TAG        0de6259a3edfd5cc041ec42b2148599ce63e73cb # v0.11.1
        GIT_SUBMODULES Clarabel.rs
        SOURCE_SUBDIR  geoqp-do-not-add
        EXCLUDE_FROM_ALL)

    FetchContent_MakeAvailable(daqp piqp clarabel)
    set(GEOQP_BENCH_HAVE_DAQP ON)
    set(GEOQP_BENCH_HAVE_PIQP ON)

    find_program(CARGO_EXECUTABLE cargo)
    if(CARGO_EXECUTABLE)
        find_package(Threads REQUIRED)
        set(CLARABEL_CARGO_TARGET_DIR ${CMAKE_BINARY_DIR}/clarabel-rust)
        set(CLARABEL_STATIC_LIB ${CLARABEL_CARGO_TARGET_DIR}/release/libclarabel_c.a)
        set(CLARABEL_LOCKFILE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/clarabel/Cargo.lock)
        add_custom_command(
            OUTPUT  ${CLARABEL_STATIC_LIB}
            COMMAND ${CMAKE_COMMAND} -E copy ${CLARABEL_LOCKFILE} Cargo.lock
            COMMAND ${CMAKE_COMMAND} -E env
                    CARGO_HOME=${CMAKE_BINARY_DIR}/cargo-home
                    CARGO_TARGET_DIR=${CLARABEL_CARGO_TARGET_DIR}
                    ${CARGO_EXECUTABLE} build --release --locked
            WORKING_DIRECTORY ${clarabel_SOURCE_DIR}/rust_wrapper
            DEPENDS ${CLARABEL_LOCKFILE}
            COMMENT "Building Clarabel Rust core (cargo)"
            VERBATIM)
        add_custom_target(clarabel_rust DEPENDS ${CLARABEL_STATIC_LIB})

        add_library(geoqp_bench_clarabel INTERFACE)
        target_include_directories(geoqp_bench_clarabel INTERFACE ${clarabel_SOURCE_DIR}/include)
        target_link_libraries(geoqp_bench_clarabel INTERFACE
            ${CLARABEL_STATIC_LIB} Threads::Threads ${CMAKE_DL_LIBS} m)
        add_dependencies(geoqp_bench_clarabel clarabel_rust)
        set(GEOQP_BENCH_HAVE_CLARABEL ON)
    else()
        message(STATUS "cargo not found: benchmarks built without Clarabel")
    endif()
endif()

# ─────────────────────────────────────────────────────────────────────────────
# Shared solver runners (one cold solve per call, uniform timing and checks)
# ─────────────────────────────────────────────────────────────────────────────
add_library(geoqp_bench_solvers STATIC solver_suite.cpp)
target_include_directories(geoqp_bench_solvers PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR} ${qpOASES_INCLUDE_DIR})
target_link_libraries(geoqp_bench_solvers PUBLIC geoqp Eigen3::Eigen ${qpOASES_LIBRARY})
target_compile_features(geoqp_bench_solvers PUBLIC cxx_std_17)

if(osqp_FOUND)
    target_link_libraries(geoqp_bench_solvers PRIVATE osqp::osqp)
    target_compile_definitions(geoqp_bench_solvers PRIVATE HAVE_OSQP=1)
endif()
if(GEOQP_BENCH_HAVE_DAQP)
    target_link_libraries(geoqp_bench_solvers PRIVATE daqp::daqpstat)
    target_compile_definitions(geoqp_bench_solvers PRIVATE HAVE_DAQP=1)
endif()
if(GEOQP_BENCH_HAVE_PIQP)
    target_link_libraries(geoqp_bench_solvers PRIVATE piqp_header_only)
    target_compile_definitions(geoqp_bench_solvers PRIVATE HAVE_PIQP=1)
endif()
if(GEOQP_BENCH_HAVE_CLARABEL)
    target_link_libraries(geoqp_bench_solvers PRIVATE geoqp_bench_clarabel)
    target_compile_definitions(geoqp_bench_solvers PRIVATE HAVE_CLARABEL=1)
endif()

add_executable(check_solver_suite check_solver_suite.cpp)
target_link_libraries(check_solver_suite PRIVATE geoqp_bench_solvers)
if(GEOQP_BUILD_TESTS)
    add_test(NAME solver_suite_agrees COMMAND check_solver_suite)
endif()

# ─────────────────────────────────────────────────────────────────────────────
# Benchmarks
# ─────────────────────────────────────────────────────────────────────────────
message(STATUS "Building box-focused benchmark")
add_executable(benchmark_box benchmark-box.cpp)
target_link_libraries(benchmark_box PRIVATE geoqp_bench_solvers)
install(TARGETS benchmark_box RUNTIME DESTINATION bin)

if(OsqpEigen_FOUND)
    message(STATUS "Building LIPM walking controller benchmark")
    add_executable(benchmark_lipm_walking benchmark_lipm_walking.cpp)
    target_link_libraries(benchmark_lipm_walking PRIVATE geoqp_bench_solvers OsqpEigen::OsqpEigen)
    if(QpSolversEigen_FOUND)
        target_link_libraries(benchmark_lipm_walking PRIVATE QpSolversEigen::QpSolversEigen)
    endif()
    install(TARGETS benchmark_lipm_walking RUNTIME DESTINATION bin)
else()
    message(STATUS "Skipping LIPM walking benchmark (requires OSQP)")
endif()
