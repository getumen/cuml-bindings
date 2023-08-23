add_library(objcuml4c OBJECT)

set(CUML_VERSION v21.08.00)

find_package(CUDA REQUIRED)

if(CUDA_FOUND)
    message("cuda version: " ${CUDA_VERSION})

    # Thrust
    find_package(Thrust REQUIRED)
    thrust_create_target(Thrust)

    # RMM
    find_package(rmm REQUIRED)

    # RAFT
    find_package(raft REQUIRED)

    target_compile_options(objcuml4c PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>)
    set_target_properties(objcuml4c
    PROPERTIES CXX_STANDARD                        17
            CXX_STANDARD_REQUIRED               ON
            CUDA_STANDARD                       17
            CUDA_STANDARD_REQUIRED              ON
            POSITION_INDEPENDENT_CODE           ON
            INTERFACE_POSITION_INDEPENDENT_CODE ON)


    find_package(cuml REQUIRED)

    target_link_libraries(objcuml4c PRIVATE 
        ${CUDA_LIBRARIES} 
        Thrust 
        rmm::rmm 
        treelite 
        cuml 
        cuml++ 
        cumlprims)

    target_include_directories(objcuml4c PUBLIC 
        ${PROJECT_SOURCE_DIR}/include
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )

    target_sources(objcuml4c PRIVATE
        agglomerative_clustering.cu
        async_utils.cuh
        cuda_utils.cu
        cuda_utils.h
        dbscan.cu
        device_allocator.cu
        device_allocator.h
        fil_utils.cu
        fil_utils.h
        fil.cu
        handle_utils.cu
        handle_utils.h
        kmeans.cu
        preprocessor.h
        stream_allocator.cu
        stream_allocator.h
        treelite_utils.cuh
        unique_marker.cuh
    )

endif()
