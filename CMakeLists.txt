cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 61)
endif()

project(cuml4c VERSION 1.0 LANGUAGES CXX CUDA)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

enable_testing()

add_subdirectory(src)

add_library(cuml4c SHARED)
target_link_libraries(cuml4c PUBLIC objcuml4c)

add_subdirectory(tests)

install(TARGETS cuml4c
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION include)

install(DIRECTORY include/cuml4c
  DESTINATION include)