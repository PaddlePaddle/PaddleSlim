CMAKE_MINIMUM_REQUIRED(VERSION 3.2)

project(mkldnn_quantaware_demo CXX C)
set(DEMO_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(DEMO_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

option(USE_GPU      "Compile the inference code with the support CUDA GPU" OFF)
option(USE_PROFILER "Whether enable Paddle's profiler." OFF)

set(USE_SHARED OFF)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
if(NOT PADDLE_ROOT)
  set(PADDLE_ROOT ${DEMO_SOURCE_DIR}/fluid_inference)
endif()
find_package(Fluid)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11")

if(USE_PROFILER)
    find_package(Gperftools REQUIRED)
    include_directories(${GPERFTOOLS_INCLUDE_DIR})
    add_definitions(-DWITH_GPERFTOOLS)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if(PADDLE_FOUND)
  add_executable(inference sample_tester.cc)
  target_link_libraries(inference
      ${PADDLE_LIBRARIES}
      ${PADDLE_THIRD_PARTY_LIBRARIES}
      rt dl pthread)
  if (mklml_FOUND)
    target_link_libraries(inference "-L${THIRD_PARTY_ROOT}/install/mklml/lib -liomp5 -Wl,--as-needed")
  endif()
else()
  message(FATAL_ERROR "Cannot find PaddlePaddle Fluid under ${PADDLE_ROOT}")
endif()
