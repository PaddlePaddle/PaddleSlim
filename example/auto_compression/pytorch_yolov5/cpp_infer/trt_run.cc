#include <chrono>
#include <iostream>
#include <memory>
#include <numeric>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <cuda_runtime.h>

#include "paddle/include/paddle_inference_api.h"
#include "paddle/include/experimental/phi/common/float16.h"

using paddle_infer::Config;
using paddle_infer::Predictor;
using paddle_infer::CreatePredictor;
using paddle_infer::PrecisionType;
using phi::dtype::float16;

DEFINE_string(model_dir, "", "Directory of the inference model.");
DEFINE_string(model_file, "", "Path of the inference model file.");
DEFINE_string(params_file, "", "Path of the inference params file.");
DEFINE_string(run_mode, "trt_fp32", "run_mode which can be: trt_fp32, trt_fp16 and trt_int8");
DEFINE_int32(batch_size, 1, "Batch size.");
DEFINE_int32(gpu_id, 0, "GPU card ID num.");
DEFINE_int32(trt_min_subgraph_size, 3, "tensorrt min_subgraph_size");
DEFINE_int32(warmup, 50, "warmup");
DEFINE_int32(repeats, 1000, "repeats");

using Time = decltype(std::chrono::high_resolution_clock::now());
Time time() { return std::chrono::high_resolution_clock::now(); };
double time_diff(Time t1, Time t2) {
  typedef std::chrono::microseconds ms;
  auto diff = t2 - t1;
  ms counter = std::chrono::duration_cast<ms>(diff);
  return counter.count() / 1000.0;
}

std::shared_ptr<Predictor> InitPredictor() {
  Config config;
  std::string model_path;
  if (FLAGS_model_dir != "") {
    config.SetModel(FLAGS_model_dir);
    model_path = FLAGS_model_dir.substr(0, FLAGS_model_dir.find_last_of("/"));
  } else {
    config.SetModel(FLAGS_model_file, FLAGS_params_file);
    model_path = FLAGS_model_file.substr(0, FLAGS_model_file.find_last_of("/"));
  }
  // enable tune
  std::cout << "model_path: " << model_path << std::endl;
  config.EnableUseGpu(256, FLAGS_gpu_id);
  if (FLAGS_run_mode == "trt_fp32") {
    config.EnableTensorRtEngine(1 << 30, FLAGS_batch_size, FLAGS_trt_min_subgraph_size,
                                PrecisionType::kFloat32, false, false);
  } else if (FLAGS_run_mode == "trt_fp16") {
    config.EnableTensorRtEngine(1 << 30, FLAGS_batch_size, FLAGS_trt_min_subgraph_size,
                                PrecisionType::kHalf, false, false);
  } else if (FLAGS_run_mode == "trt_int8") {
    config.EnableTensorRtEngine(1 << 30, FLAGS_batch_size, FLAGS_trt_min_subgraph_size,
                                PrecisionType::kInt8, false, false);
  }
  config.EnableMemoryOptim();
  config.SwitchIrOptim(true);
  return CreatePredictor(config);
}

template <typename type>
void run(Predictor *predictor, const std::vector<type> &input,
         const std::vector<int> &input_shape, type* out_data, std::vector<int> out_shape) {
  
    // prepare input
    int input_num = std::accumulate(input_shape.begin(), input_shape.end(), 1,
                                    std::multiplies<int>());
    
    auto input_names = predictor->GetInputNames();
    auto input_t = predictor->GetInputHandle(input_names[0]);
    input_t->Reshape(input_shape);
    input_t->CopyFromCpu(input.data());
  
  for (int i = 0; i < FLAGS_warmup; ++i)
    CHECK(predictor->Run());

  auto st = time();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    auto input_names = predictor->GetInputNames();
    auto input_t = predictor->GetInputHandle(input_names[0]);
    input_t->Reshape(input_shape);
    input_t->CopyFromCpu(input.data());

    CHECK(predictor->Run());

    auto output_names = predictor->GetOutputNames();
    auto output_t = predictor->GetOutputHandle(output_names[0]);
    std::vector<int> output_shape = output_t->shape();
    output_t -> ShareExternalData<type>(out_data, out_shape, paddle_infer::PlaceType::kGPU);
  }
  
  LOG(INFO) << "[" << FLAGS_run_mode << " bs-" << FLAGS_batch_size << " ] run avg time is " << time_diff(st, time()) / FLAGS_repeats
            << " ms";
}

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  auto predictor = InitPredictor();
  std::vector<int> input_shape = {FLAGS_batch_size, 3, 640, 640};
  // float16
  using dtype = float16;
  std::vector<dtype> input_data(FLAGS_batch_size * 3 * 640 * 640, dtype(1.0));

  dtype *out_data;
  int out_data_size = FLAGS_batch_size * 25200 * 85;
  cudaHostAlloc((void**)&out_data, sizeof(float) * out_data_size, cudaHostAllocMapped);

  std::vector<int> out_shape{ FLAGS_batch_size, 1, 25200, 85};
  run<dtype>(predictor.get(), input_data, input_shape, out_data, out_shape);
  return 0;
}
