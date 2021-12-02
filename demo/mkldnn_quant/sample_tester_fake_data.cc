/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <paddle_inference_api.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

DEFINE_string(infer_model, "", "path to the model");
DEFINE_string(infer_data, "", "path to the input data");
DEFINE_int32(batch_size, 1, "inference batch size");
DEFINE_int32(iterations,
             0,
             "number of batches to process. 0 means testing whole dataset");
DEFINE_int32(num_threads, 1, "num of threads to run in parallel");
DEFINE_bool(with_accuracy_layer,
            true,
            "Set with_accuracy_layer to true if provided model has accuracy "
            "layer and requires label input");
DEFINE_bool(use_analysis,
            false,
            "If use_analysis is set to true, the model will be optimized");
DEFINE_int32(warmup_iter, 2, "number of warmup batches");            

struct Timer {
  std::chrono::high_resolution_clock::time_point start;
  std::chrono::high_resolution_clock::time_point startu;

  void tic() { start = std::chrono::high_resolution_clock::now(); }
  double toc() {
    startu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(startu -
                                                                  start);
    double used_time_ms = static_cast<double>(time_span.count()) * 1000.0;
    return used_time_ms;
  }
};

static void SetIrOptimConfig(paddle::AnalysisConfig *cfg) {
  cfg->DisableGpu();
  cfg->SwitchIrOptim();
  cfg->EnableMKLDNN();
}

int main(int argc, char *argv[]) {
  // InitFLAGS(argc, argv);
  google::InitGoogleLogging(*argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  paddle::AnalysisConfig cfg;
  cfg.SetModel(FLAGS_infer_model);
  cfg.SetCpuMathLibraryNumThreads(FLAGS_num_threads);
  if (FLAGS_use_analysis) {
    SetIrOptimConfig(&cfg);
  }
  cfg.SwitchUseFeedFetchOps(false);

  auto predictor = paddle::CreatePaddlePredictor(cfg);
  int batch_size = FLAGS_batch_size;
  std::cout<<"Batch size " << FLAGS_batch_size << std::endl ;
  int channels = 3;
  int height = 224;
  int width = 224;
  int nums = batch_size * channels * height * width;
  auto input_shape = {batch_size, channels, height, width};
  float* input = new float[nums];
  for (int i = 0; i < nums; ++i) input[i] = 0;
  auto input_names = predictor->GetInputNames();

  auto input_t = predictor->GetInputTensor(input_names[0]);
  input_t->Reshape(input_shape);
  input_t->copy_from_cpu<float>(input);
  
  for (auto iter = 0; iter<2; iter++){
    predictor->ZeroCopyRun();
    LOG(INFO) <<"Warmup " << iter << " batches";
  }
  Timer run_timer;
  double elapsed_time=0;
  run_timer.tic();
  FLAGS_iterations= (FLAGS_iterations==0) ? (5000/FLAGS_batch_size) : FLAGS_iterations;
  for (auto iter = 0; iter<FLAGS_iterations; iter++){
    predictor->ZeroCopyRun();
  }

  LOG(INFO) <<"Iterations executed are " << FLAGS_iterations;
  elapsed_time+=run_timer.toc();
  auto batch_latency = elapsed_time / FLAGS_iterations;
  auto sample_latency = batch_latency / FLAGS_batch_size;
  // How to calculate fps. Using 1000.f/amounts ?
  std::cout<<"Batch_latency: " << batch_latency << std::endl;
  std::cout<<"Sample_latency: " << sample_latency << std::endl;
  std::cout<<"FPS: " << 1000.f/sample_latency << std::endl;
}
