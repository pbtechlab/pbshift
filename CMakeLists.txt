cmake_minimum_required(VERSION 3.20)
project(pbshift CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(PFFFT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/pffft)

add_library(pffft_static STATIC
  ${PFFFT_DIR}/src/pffft.c
  ${PFFFT_DIR}/src/pffft_common.c)
target_include_directories(pffft_static PUBLIC ${PFFFT_DIR}/include/pffft)
target_compile_definitions(pffft_static PUBLIC PFFFT_STATIC_DEFINE)
if(NOT MSVC)
  target_compile_options(pffft_static PRIVATE -O2 -mavx)
endif()

add_library(pbshift STATIC src/pb_engine.cpp)
target_include_directories(pbshift
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(pbshift PRIVATE pffft_static)
if(NOT MSVC)
  target_compile_options(pbshift PRIVATE -O2 -Wall -Wextra)
endif()

add_executable(pbshift_cli tools/pbshift_cli/main.cpp)
target_link_libraries(pbshift_cli PRIVATE pbshift)
set_target_properties(pbshift_cli PROPERTIES
  OUTPUT_NAME pbshift
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tools/bin)
if(NOT MSVC)
  target_link_options(pbshift_cli PRIVATE -static)
endif()

# Standalone multi-resolution offline renderer (header-only engine + pffft).
add_executable(multires_cli tools/multires_cli/main.cpp)
target_include_directories(multires_cli PRIVATE src include)
target_link_libraries(multires_cli PRIVATE pffft_static)
set_target_properties(multires_cli PROPERTIES
  OUTPUT_NAME multires
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tools/bin)
if(NOT MSVC)
  target_link_options(multires_cli PRIVATE -static)
endif()

enable_testing()
add_executable(test_core tests/test_core.cpp)
target_include_directories(test_core PRIVATE src include)
target_link_libraries(test_core PRIVATE pffft_static)
add_test(NAME core COMMAND test_core)

add_executable(test_streaming tests/test_streaming.cpp)
target_link_libraries(test_streaming PRIVATE pbshift)
add_test(NAME streaming COMMAND test_streaming)

add_executable(rt_bench tests/rt_bench.cpp)
target_link_libraries(rt_bench PRIVATE pbshift)

add_executable(realtime_insert examples/realtime_insert.cpp)
target_link_libraries(realtime_insert PRIVATE pbshift)
add_test(NAME realtime_insert COMMAND realtime_insert)
