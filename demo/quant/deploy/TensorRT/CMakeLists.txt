cmake_minimum_required(VERSION 3.0)
project(inference_test CXX C)
option(WITH_MKL        "Compile demo with MKL/OpenBlas support, default use MKL."       OFF)
option(WITH_GPU        "Compile demo with GPU/CPU, default use CPU."                    OFF)
option(WITH_STATIC_LIB "Compile demo with static/shared library, default use static."   OFF)
option(USE_TENSORRT "Compile demo with TensorRT."   OFF)

if(NOT DEFINED PADDLE_LIB)
  message(FATAL_ERROR "please set PADDLE_LIB with -DPADDLE_LIB=/path/paddle/lib")
endif()

# check file system
file(READ "/etc/issue" ETC_ISSUE)
string(REGEX MATCH "Debian|Ubuntu|CentOS" DIST ${ETC_ISSUE})

if(DIST STREQUAL "Debian")
  message(STATUS ">>>> Found Debian <<<<")
elseif(DIST STREQUAL "Ubuntu")
  message(STATUS ">>>> Found Ubuntu <<<<")
elseif(DIST STREQUAL "CentOS")
  message(STATUS ">>>> Found CentOS <<<<")
else()
  message(STATUS ">>>> Found unknown distribution <<<<")
endif()

include_directories("${PADDLE_LIB}/")
set(PADDLE_LIB_THIRD_PARTY_PATH "${PADDLE_LIB}/third_party/install/")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}protobuf/include")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}glog/include")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}gflags/include")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}xxhash/include")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}zlib/include")
include_directories("${PADDLE_LIB}/third_party/boost")
include_directories("${PADDLE_LIB}/third_party/eigen3")
include_directories("${PADDLE_LIB}/paddle/include")
include_directories("${PADDLE_LIB}/paddle")

link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}zlib/lib")
link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}protobuf/lib")
link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}glog/lib")
link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}gflags/lib")
link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}xxhash/lib")
link_directories("${PADDLE_LIB}/paddle/lib")


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -std=c++11")
message("flags" ${CMAKE_CXX_FLAGS})

if (USE_TENSORRT AND WITH_GPU)
  message("=====> TENSORRT_INCLUDE_DIR is ${TENSORRT_INCLUDE_DIR}")
  message("=====> TENSORRT_LIB_DIR is ${TENSORRT_LIB_DIR}")
  include_directories("${TENSORRT_INCLUDE_DIR}")
  link_directories("${TENSORRT_LIB_DIR}")
endif()

if(WITH_MKL)
  set(MATH_LIB_PATH "${PADDLE_LIB_THIRD_PARTY_PATH}mklml")
  include_directories("${MATH_LIB_PATH}/include")
  set(MATH_LIB ${MATH_LIB_PATH}/lib/libmklml_intel${CMAKE_SHARED_LIBRARY_SUFFIX}
                 ${MATH_LIB_PATH}/lib/libiomp5${CMAKE_SHARED_LIBRARY_SUFFIX})
  set(MKLDNN_PATH "${PADDLE_LIB_THIRD_PARTY_PATH}mkldnn")
  if(EXISTS ${MKLDNN_PATH})
    include_directories("${MKLDNN_PATH}/include")
    set(MKLDNN_LIB ${MKLDNN_PATH}/lib/libmkldnn.so.0)
  endif()
else()
  set(MATH_LIB ${PADDLE_LIB_THIRD_PARTY_PATH}openblas/lib/libopenblas${CMAKE_STATIC_LIBRARY_SUFFIX})
endif()

# Note: libpaddle_inference_api.so/a must put before libpaddle_fluid.so/a
if(WITH_STATIC_LIB)
  set(DEPS ${PADDLE_LIB}/paddle/lib/libpaddle_fluid${CMAKE_STATIC_LIBRARY_SUFFIX})
else()
  set(DEPS ${PADDLE_LIB}/paddle/lib/libpaddle_fluid${CMAKE_SHARED_LIBRARY_SUFFIX})
endif()

set(EXTERNAL_LIB "-lrt -ldl -lpthread -lprotobuf")
set(DEPS ${DEPS}
  ${MATH_LIB} ${MKLDNN_LIB}
  glog gflags protobuf z xxhash
  ${EXTERNAL_LIB})

if(WITH_GPU)
    if (USE_TENSORRT)
      set(DEPS ${DEPS} ${TENSORRT_LIB_DIR}/libnvinfer${CMAKE_SHARED_LIBRARY_SUFFIX})
      set(DEPS ${DEPS} ${TENSORRT_LIB_DIR}/libnvinfer_plugin${CMAKE_SHARED_LIBRARY_SUFFIX})
    endif()
    set(DEPS ${DEPS} $ENV{CUDA_LIB})
endif()

set(TestFiles "trt_clas_benchmark.cc";
              "trt_gen_calib_table_test.cc";
              "slim_clas_test.cc";)

foreach(testsourcefile ${TestFiles})
  message("====> ${testsourcefile} will be compiled")
  # add executable for all test files
  string(REPLACE ".cc" "" testname ${testsourcefile})
  add_executable(${testname} ${testsourcefile})

  # link libs
  target_link_libraries(${testname} ${DEPS})
  
endforeach()
