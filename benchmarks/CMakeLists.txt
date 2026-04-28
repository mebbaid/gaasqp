find_package(QpSolversEigen QUIET)
find_package(OsqpEigen QUIET)
find_package(qpOASES QUIET)

if(qpOASES_FOUND)
    message(STATUS "Building box-focused benchmark")

    add_executable(benchmark_box benchmark-box.cpp)
    target_include_directories(benchmark_box PRIVATE ${qpOASES_INCLUDE_DIR})
    if(QpSolversEigen_FOUND AND OsqpEigen_FOUND)
        target_link_libraries(benchmark_box PRIVATE
            geoqp
            QpSolversEigen::QpSolversEigen
            OsqpEigen::OsqpEigen
            ${qpOASES_LIBRARY})
        target_compile_definitions(benchmark_box PRIVATE HAVE_OSQP=1)
    elseif(QpSolversEigen_FOUND)
        target_link_libraries(benchmark_box PRIVATE
            geoqp
            QpSolversEigen::QpSolversEigen
            ${qpOASES_LIBRARY})
    else()
        target_link_libraries(benchmark_box PRIVATE
            geoqp
            ${qpOASES_LIBRARY})
    endif()
    target_compile_features(benchmark_box PRIVATE cxx_std_17)

    install(TARGETS benchmark_box
        RUNTIME DESTINATION bin)
endif()


if(OsqpEigen_FOUND AND qpOASES_FOUND)
    message(STATUS "Building LIPM walking controller benchmark")
    
    add_executable(benchmark_lipm_walking benchmark_lipm_walking.cpp)
    target_include_directories(benchmark_lipm_walking PRIVATE ${qpOASES_INCLUDE_DIR})
    target_link_libraries(benchmark_lipm_walking PRIVATE 
        geoqp 
        OsqpEigen::OsqpEigen
        ${qpOASES_LIBRARY})
    if(QpSolversEigen_FOUND)
        target_link_libraries(benchmark_lipm_walking PRIVATE QpSolversEigen::QpSolversEigen)
    endif()
    target_compile_features(benchmark_lipm_walking PRIVATE cxx_std_17)
    
    install(TARGETS benchmark_lipm_walking
        RUNTIME DESTINATION bin)
else()
    message(STATUS "Skipping LIPM walking benchmark (requires both OSQP and qpOASES)")
endif()
