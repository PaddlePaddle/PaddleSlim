#include <numeric>
#include <iostream>
#include <memory>
#include <chrono>
#include <random>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "paddle/include/paddle_inference_api.h"

using paddle::AnalysisConfig;

DEFINE_string(model_file, "", "Path of the inference model file.");
DEFINE_string(params_file, "", "Path of the inference params file.");
DEFINE_string(model_dir, "", "Directory of the inference model.");
DEFINE_int32(batch_size, 1, "Batch size.");

float Random(float low, float high) {
  static std::random_device rd;
  static std::mt19937 mt(rd());
  std::uniform_real_distribution<double> dist(low, high);
  return dist(mt);
}

using Time = decltype(std::chrono::high_resolution_clock::now());
Time time() { return std::chrono::high_resolution_clock::now(); };
double time_diff(Time t1, Time t2) {
  typedef std::chrono::microseconds ms;
  auto diff = t2 - t1;
  ms counter = std::chrono::duration_cast<ms>(diff);
  return counter.count() / 1000.0;
}

std::unique_ptr<paddle::PaddlePredictor> CreatePredictor() {
  AnalysisConfig config;
  if (FLAGS_model_dir != "") {
    config.SetModel(FLAGS_model_dir);
  } else {
    config.SetModel(FLAGS_model_file,
                    FLAGS_params_file);
  }
  config.EnableUseGpu(500, 0);
  // We use ZeroCopy, so we set config.SwitchUseFeedFetchOps(false) here.
  config.SwitchUseFeedFetchOps(false);
  config.EnableTensorRtEngine(1 << 30, FLAGS_batch_size, 5, AnalysisConfig::Precision::kInt8, false, true /*use_calib*/);
  return CreatePaddlePredictor(config);
}

void run(paddle::PaddlePredictor *predictor,
         std::vector<float>& input,
         const std::vector<int>& input_shape, 
         std::vector<float> *out_data) {
  int input_num = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<int>());
  for (size_t i = 0; i < 500; i++) {
    // We use random data here for example. Change this to real data in your application. 
    for (int j = 0; j < input_num; j++) {
      input[j] = Random(0, 1.0);
    }
    auto input_names = predictor->GetInputNames();
    auto input_t = predictor->GetInputTensor(input_names[0]);
    input_t->Reshape(input_shape);
    input_t->copy_from_cpu(input.data());

    // Run predictor to generate calibration table. Can be very time-consuming.     
    CHECK(predictor->ZeroCopyRun());
  }
}

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  auto predictor = CreatePredictor();
  std::vector<int> input_shape = {FLAGS_batch_size, 3, 224, 224}; 
  // Init input as 1.0 here for example. You can also load preprocessed real pictures to vectors as input.
  std::vector<float> input_data(FLAGS_batch_size * 3 * 224 * 224, 1.0); 
  std::vector<float> out_data;
  run(predictor.get(), input_data, input_shape, &out_data);
  return 0;
}

