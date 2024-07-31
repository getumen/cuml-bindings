cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

add_library(objcuml4c OBJECT)

find_package(CUDA REQUIRED)
find_package(rmm 24.06.00)
find_package(raft 24.06.00)

if(CUDA_FOUND)
    message("cuda version: " ${CUDA_VERSION})

    target_compile_options(objcuml4c PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>)
    set_target_properties(objcuml4c
    PROPERTIES CXX_STANDARD                        17
            CXX_STANDARD_REQUIRED               ON
            CUDA_STANDARD                       17
            CUDA_STANDARD_REQUIRED              ON
            POSITION_INDEPENDENT_CODE           ON
            INTERFACE_POSITION_INDEPENDENT_CODE ON)

    target_link_libraries(objcuml4c PRIVATE 
        ${CUDA_LIBRARIES} 
        treelite
        fmt
        cuml 
        cuml++ 
        cumlprims_mg
        rmm::rmm
        raft::raft)

    target_include_directories(objcuml4c PUBLIC 
        ${PROJECT_SOURCE_DIR}/include
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )

    target_sources(objcuml4c PRIVATE
        agglomerative_clustering.cu
        dbscan.cu
        device_resource_handle.cu
        fil.cu
        kmeans.cu
        linear_regression.cu
        memory_resource.cu
    )

endif()
