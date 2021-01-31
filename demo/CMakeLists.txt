project(mkldnn_int8_demo CXX C)
option(WITH_MKL        "Compile demo with MKL/OpenBlas support, default use MKL."       ON)
option(WITH_GPU        "Compile demo with GPU/CPU, default use CPU."                    OFF)
option(WITH_STATIC_LIB "Compile demo with static/shared library, default use static."   ON)


macro(safe_set_static_flag)
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
endmacro()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g")
set(CMAKE_STATIC_LIBRARY_PREFIX "")
message("flags" ${CMAKE_CXX_FLAGS})

if(NOT DEFINED PADDLE_LIB)
  message(FATAL_ERROR "please set PADDLE_LIB with -DPADDLE_LIB=/path/paddle/lib")
endif()
if(NOT DEFINED DEMO_NAME)
  message(FATAL_ERROR "please set DEMO_NAME with -DDEMO_NAME=demo_name")
endif()

# # Support directory orgnizations
# find_path(PADDLE_INC_DIR NAMES paddle_inference_api.h PATHS ${PADDLE_ROOT}/paddle/include)
# if(PADDLE_INC_DIR)
#   set(LIB_PATH "paddle/lib")
# else()
#   find_path(PADDLE_INC_DIR NAMES paddle/fluid/inference/paddle_inference_api.h PATHS ${PADDLE_ROOT})
#   if(PADDLE_INC_DIR)
#     include_directories(${PADDLE_ROOT}/paddle/fluid/inference)
#   endif()
#   set(LIB_PATH "paddle/fluid/inference")
# endif()
  
# include_directories(${PADDLE_INC_DIR})

#WHERE TO LOOK FOR HEADERS
include_directories("${PADDLE_LIB}")
include_directories("${PADDLE_LIB}/paddle/include")
include_directories("${PADDLE_LIB}/third_party/install/protobuf/include")
include_directories("${PADDLE_LIB}/third_party/install/glog/include")
include_directories("${PADDLE_LIB}/third_party/install/gflags/include")
include_directories("${PADDLE_LIB}/third_party/install/xxhash/include")
include_directories("${PADDLE_LIB}/third_party/install/zlib/include")
include_directories("${PADDLE_LIB}/third_party/boost")
include_directories("${PADDLE_LIB}/third_party/eigen3")

link_directories("${PADDLE_LIB}/third_party/install/zlib/lib")
# WHERE TO LOOK FOR LIBRARIES
link_directories("${PADDLE_LIB}/third_party/install/protobuf/lib")
link_directories("${PADDLE_LIB}/third_party/install/glog/lib")
link_directories("${PADDLE_LIB}/third_party/install/gflags/lib")
link_directories("${PADDLE_LIB}/third_party/install/xxhash/lib")
link_directories("${PADDLE_LIB}/paddle/lib")

add_executable(${DEMO_NAME} ${DEMO_NAME}.cc)

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
  set(MATH_LIB ${PADDLE_LIB}/third_party/install/openblas/lib/libopenblas${CMAKE_STATIC_LIBRARY_SUFFIX})
endif()

# Note: libpaddle_inference_api.so/a must put before libpaddle_fluid.so/a
if(WITH_STATIC_LIB)
  set(DEPS
      ${PADDLE_LIB}/paddle/lib/libpaddle_fluid${CMAKE_STATIC_LIBRARY_SUFFIX})
else()
  set(DEPS
      ${PADDLE_LIB}/paddle/lib/libpaddle_fluid${CMAKE_SHARED_LIBRARY_SUFFIX})
endif()

# set(protobuflib ${PADDLE_LIB}/third_party/install/protobuf/lib/libprotobuf${CMAKE_STATIC_LIBRARY_SUFFIX})
# set(gloglib ${PADDLE_LIB}/third_party/install/glog/lib/libglog${CMAKE_STATIC_LIBRARY_SUFFIX})
# set(gflagslib ${PADDLE_LIB}/third_party/install/gflags/lib/libgflags${CMAKE_STATIC_LIBRARY_SUFFIX})
# set(xxhashlib ${PADDLE_LIB}/third_party/install/xxhash/lib/libxxhash${CMAKE_STATIC_LIBRARY_SUFFIX})

set(EXTERNAL_LIB "-lrt -ldl -lpthread")
set(DEPS ${DEPS}
    ${MATH_LIB} ${MKLDNN_LIB} glog gflags xxhash protobuf z 
    ${EXTERNAL_LIB})
    # ${gloglib} ${gflagslib} ${protobuflib} ${xxhashlib}
if(WITH_GPU)
  if (USE_TENSORRT)
    set(DEPS ${DEPS}
        ${TENSORRT_ROOT}/lib/libnvinfer${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(DEPS ${DEPS}
        ${TENSORRT_ROOT}/lib/libnvinfer_plugin${CMAKE_SHARED_LIBRARY_SUFFIX})
  endif()
  set(DEPS ${DEPS} ${CUDA_LIB}/libcudart${CMAKE_SHARED_LIBRARY_SUFFIX})
  set(DEPS ${DEPS} ${CUDA_LIB}/libcudart${CMAKE_SHARED_LIBRARY_SUFFIX} )
  set(DEPS ${DEPS} ${CUDA_LIB}/libcublas${CMAKE_SHARED_LIBRARY_SUFFIX} )
  set(DEPS ${DEPS} ${CUDNN_LIB}/libcudnn${CMAKE_SHARED_LIBRARY_SUFFIX} )
endif()

target_link_libraries(${DEMO_NAME} ${DEPS})
