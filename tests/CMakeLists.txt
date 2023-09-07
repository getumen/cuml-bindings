cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(
  cuml_test
  fil_test.cpp
)

target_compile_options(cuml_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>)
set_target_properties(cuml_test
        PROPERTIES CXX_STANDARD             17
        CXX_STANDARD_REQUIRED               ON
        CUDA_STANDARD                       17
        CUDA_STANDARD_REQUIRED              ON
        POSITION_INDEPENDENT_CODE           ON
        INTERFACE_POSITION_INDEPENDENT_CODE ON)

target_link_libraries(
    cuml_test
    ${CUDA_LIBRARIES} 
    treelite 
    cuml 
    cuml++ 
    cumlprims_mg
    cuml4c
    gtest_main
)

target_include_directories(cuml_test PUBLIC 
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${GTEST_INCLUDE_DIRS}
)

file(COPY ${CMAKE_SOURCE_DIR}/testdata DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

include(GoogleTest)
gtest_discover_tests(cuml_test)
