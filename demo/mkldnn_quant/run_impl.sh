LIB_DIR=$1
CPP_NAME=$2
MODEL_FILE_DIR=$3
INPUT_PATH=$4
WITH_MKL=$5

mkdir -p build_${CPP_NAME}
cd build_${CPP_NAME}
rm -rf *

cmake .. -DPADDLE_LIB=${LIB_DIR} \
  -DWITH_MKL=${WITH_MKL} \
  -DDEMO_NAME=${CPP_NAME} \
  -DWITH_STATIC_LIB=OFF \

make -j

./${CPP_NAME} --model_path=${MODEL_FILE_DIR} --input_path=${INPUT_PATH}
