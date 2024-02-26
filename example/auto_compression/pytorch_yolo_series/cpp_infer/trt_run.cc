// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <chrono>
#include <iostream>
#include <memory>
#include <numeric>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "paddle_inference_api.h"

using paddle_infer::Config;
using paddle_infer::CreatePredictor;
using paddle_infer::PrecisionType;
using paddle_infer::Predictor;

DEFINE_string(model_dir, "", "Directory of the inference model.");
DEFINE_string(model_file, "", "Path of the inference model file.");
DEFINE_string(params_file, "", "Path of the inference params file.");
DEFINE_string(
    run_mode,
    "paddle_gpu",
    "run_mode which can be: trt_fp32, trt_fp16 and trt_int8 and paddle_gpu");
DEFINE_int32(batch_size, 1, "Batch size.");
DEFINE_int32(gpu_id, 0, "GPU card ID num.");
DEFINE_int32(trt_min_subgraph_size, 3, "tensorrt min_subgraph_size");
DEFINE_int32(warmup, 50, "warmup");
DEFINE_int32(repeats, 1000, "repeats");
DEFINE_bool(use_dynamic_shape, false, "use trt dynaminc shape.");
DEFINE_bool(use_calib, true, "use trt int8 calibration.");

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
  if (FLAGS_model_dir != "") {
    config.SetModel(FLAGS_model_dir);
  }
  config.SetModel(FLAGS_model_file, FLAGS_params_file);

  config.EnableUseGpu(500, FLAGS_gpu_id);

  if (FLAGS_run_mode == "trt_fp32") {
    config.EnableTensorRtEngine(1 << 30 * FLAGS_batch_size,
                                FLAGS_batch_size,
                                FLAGS_trt_min_subgraph_size,
                                PrecisionType::kFloat32,
                                false,
                                false);
  } else if (FLAGS_run_mode == "trt_fp16") {
    config.EnableTensorRtEngine(1 << 30 * FLAGS_batch_size,
                                FLAGS_batch_size,
                                FLAGS_trt_min_subgraph_size,
                                PrecisionType::kHalf,
                                false,
                                false);
  } else if (FLAGS_run_mode == "trt_int8") {
    config.EnableTensorRtEngine(1 << 30 * FLAGS_batch_size,
                                FLAGS_batch_size,
                                FLAGS_trt_min_subgraph_size,
                                PrecisionType::kInt8,
                                false,
                                FLAGS_use_calib);
  }
  if (FLAGS_use_dynamic_shape) {
    std::map<std::string, std::vector<int>> min_input_shape = {
        {"image", {1, 3, 640, 640}}};
    std::map<std::string, std::vector<int>> max_input_shape = {
        {"image", {4, 3, 640, 640}}};
    std::map<std::string, std::vector<int>> opt_input_shape = {
        {"image", {2, 3, 640, 640}}};
    config.SetTRTDynamicShapeInfo(
        min_input_shape, max_input_shape, opt_input_shape);
  }
  // Open the memory optim.
  config.EnableMemoryOptim();
  config.SwitchIrDebug(true);
  config.SwitchIrOptim(true);
  return CreatePredictor(config);
}

void run(Predictor *predictor,
         const std::vector<float> &input,
         const std::vector<int> &input_shape,
         std::vector<float> *out_data) {
  int input_num = std::accumulate(
      input_shape.begin(), input_shape.end(), 1, std::multiplies<int>());

  auto input_names = predictor->GetInputNames();
  auto output_names = predictor->GetOutputNames();
  auto input_t = predictor->GetInputHandle(input_names[0]);
  input_t->Reshape(input_shape);
  input_t->CopyFromCpu(input.data());

  for (size_t i = 0; i < FLAGS_warmup; ++i) CHECK(predictor->Run());

  auto st = time();
  for (size_t i = 0; i < FLAGS_repeats; ++i) {
    CHECK(predictor->Run());
    auto output_t = predictor->GetOutputHandle(output_names[0]);
    std::vector<int> output_shape = output_t->shape();
    int out_num = std::accumulate(
        output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
    out_data->resize(out_num);
    output_t->CopyToCpu(out_data->data());
  }
  LOG(INFO) << "run avg time is " << time_diff(st, time()) / FLAGS_repeats
            << " ms";
}

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  auto predictor = InitPredictor();
  std::vector<int> input_shape = {FLAGS_batch_size, 3, 640, 640};
  std::vector<float> input_data(FLAGS_batch_size * 3 * 640 * 640);
  for (size_t i = 0; i < input_data.size(); ++i) input_data[i] = i % 255 * 0.1;
  std::vector<float> out_data;
  run(predictor.get(), input_data, input_shape, &out_data);

  return 0;
}