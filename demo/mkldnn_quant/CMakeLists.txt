project(mkldnn_int8_demo CXX C)
option(WITH_MKL        "Compile demo with MKL/OpenBlas support, default use MKL."       ON)
option(WITH_STATIC_LIB "Compile demo with static/shared library, default use static."   ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g")
set(CMAKE_STATIC_LIBRARY_PREFIX "")
message("flags" ${CMAKE_CXX_FLAGS})

if(NOT DEFINED PADDLE_LIB)
  message(FATAL_ERROR "please set PADDLE_LIB with -DPADDLE_LIB=/path/paddle/lib")
endif()
set(DEMO_NAME sample_tester)
set(DEMO_NAME_FAKE_DATA sample_tester_fake_data)
if(NOT DEFINED DEMO_NAME)
  message(FATAL_ERROR "please set DEMO_NAME with -DDEMO_NAME=demo_name")
endif()

include_directories("${PADDLE_LIB}")
include_directories("${PADDLE_LIB}/paddle/include")
include_directories("${PADDLE_LIB}/third_party/install/protobuf/include")
include_directories("${PADDLE_LIB}/third_party/install/glog/include")
include_directories("${PADDLE_LIB}/third_party/install/gflags/include")
include_directories("${PADDLE_LIB}/third_party/install/xxhash/include")
include_directories("${PADDLE_LIB}/third_party/install/cryptopp/include")
include_directories("${PADDLE_LIB}/third_party/install/utf8proc/include")
link_directories("${PADDLE_LIB}/third_party/install/protobuf/lib")
link_directories("${PADDLE_LIB}/third_party/install/glog/lib")
link_directories("${PADDLE_LIB}/third_party/install/gflags/lib")
link_directories("${PADDLE_LIB}/third_party/install/xxhash/lib")
link_directories("${PADDLE_LIB}/paddle/lib")
link_directories("${PADDLE_LIB}/third_party/install/cryptopp/lib")
link_directories("${PADDLE_LIB}/third_party/install/utf8proc/lib")

add_executable(${DEMO_NAME} ${DEMO_NAME}.cc)
add_executable(${DEMO_NAME_FAKE_DATA} ${DEMO_NAME_FAKE_DATA}.cc)

if(WITH_MKL)
  include_directories("${PADDLE_LIB}/third_party/install/mklml/include")
  set(MATH_LIB ${PADDLE_LIB}/third_party/install/mklml/lib/libmklml_intel${CMAKE_SHARED_LIBRARY_SUFFIX}
               ${PADDLE_LIB}/third_party/install/mklml/lib/libiomp5${CMAKE_SHARED_LIBRARY_SUFFIX})
  set(MKLDNN_PATH "${PADDLE_LIB}/third_party/install/mkldnn")
  if(EXISTS ${MKLDNN_PATH})
    include_directories("${MKLDNN_PATH}/include")
    set(MKLDNN_LIB ${MKLDNN_PATH}/lib/libmkldnn.so.0)
  endif()
else()
  include_directories("${PADDLE_LIB}/third_party/install/openblas/include")
  set(MATH_LIB ${PADDLE_LIB}/third_party/install/openblas/lib/libopenblas${CMAKE_STATIC_LIBRARY_SUFFIX})
endif()

# Note: libpaddle_inference_api.so/a must put before libpaddle_fluid.so/a
if(WITH_STATIC_LIB)
  set(DEPS
      ${PADDLE_LIB}/paddle/lib/libpaddle_inference${CMAKE_STATIC_LIBRARY_SUFFIX})
else()
  set(DEPS
      ${PADDLE_LIB}/paddle/lib/libpaddle_inference${CMAKE_SHARED_LIBRARY_SUFFIX})
endif()

set(EXTERNAL_LIB "-lrt -ldl -lpthread")
set(DEPS ${DEPS}
    ${MATH_LIB} ${MKLDNN_LIB} glog gflags xxhash protobuf cryptopp utf8proc
    ${EXTERNAL_LIB})

target_link_libraries(${DEMO_NAME} ${DEPS})
target_link_libraries(${DEMO_NAME_FAKE_DATA} ${DEPS})
