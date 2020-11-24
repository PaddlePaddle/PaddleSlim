#include <gflags/gflags.h>
#include <glog/logging.h>
#include <algorithm>  // std::sort
#include <numeric>  // std::iota
#include <vector>
#include <numeric>
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <chrono>
#include "paddle/include/paddle_inference_api.h"

namespace paddle {
using paddle::AnalysisConfig;

DEFINE_string(model_dir, "resnet50_quant", "Directory of the inference model.");
DEFINE_string(data_dir, "imagenet-eval-binary", "Directory of the data.");
DEFINE_bool(int8, true, "use int8 or not");

using Time = decltype(std::chrono::high_resolution_clock::now());
Time time() { return std::chrono::high_resolution_clock::now(); };
double time_diff(Time t1, Time t2) {
  typedef std::chrono::microseconds ms;
  auto diff = t2 - t1;
  ms counter = std::chrono::duration_cast<ms>(diff);
  return counter.count() / 1000.0;
}

void PrepareTRTConfig(AnalysisConfig *config, int batch_size, int id = 0) {
  int min_subgraph_size = 3;
  config->SetModel(FLAGS_model_dir + "/model", FLAGS_model_dir + "/params");
  config->EnableUseGpu(1500, id);
  // We use ZeroCopyTensor here, so we set config->SwitchUseFeedFetchOps(false)
  config->SwitchUseFeedFetchOps(false);
  config->SwitchIrDebug(true);
  if (FLAGS_int8)
      config->EnableTensorRtEngine(1 << 30, batch_size, min_subgraph_size, AnalysisConfig::Precision::kInt8, false, false);
  else
      config->EnableTensorRtEngine(1 << 30, batch_size, min_subgraph_size, AnalysisConfig::Precision::kFloat32, false, false);

}

bool test_map_cnn(int batch_size, int repeat) {
  AnalysisConfig config;
  PrepareTRTConfig(&config, batch_size);
  auto predictor = CreatePaddlePredictor(config);

  int channels = 3;
  int height = 224;
  int width = 224;
  int input_num = channels * height * width * batch_size;

  // prepare inputs
  float *input = new float[input_num];
  memset(input, 0, input_num * sizeof(float));
  float test_num = 0;
  float top1_num = 0;
  float top5_num = 0;
  std::vector<int> index(1000);
  
  for (size_t ind = 0; ind < 50000; ind++) {
    if(ind % 5000 == 0)
    LOG(INFO) << ind;
    std::ifstream fs(FLAGS_data_dir + "/" +  std::to_string(ind)  + ".data", std::ifstream::binary);
    if (!fs.is_open()) {
      LOG(FATAL) << "open input file fail.";
    }    
    auto input_data_tmp = input;
    for (int i = 0; i < input_num; ++i) {
      fs.read(reinterpret_cast<char*>(input_data_tmp), sizeof(*input_data_tmp));
      input_data_tmp++;
    }
    int label = 0;
    fs.read(reinterpret_cast<char*>(&label), sizeof(label));
    fs.close();

    auto input_names = predictor->GetInputNames();
    auto input_t = predictor->GetInputTensor(input_names[0]);
    input_t->Reshape({batch_size, channels, height, width});
    input_t->copy_from_cpu(input);

    CHECK(predictor->ZeroCopyRun());
    std::vector<float> out_data;
    auto output_names = predictor->GetOutputNames();
    auto output_t = predictor->GetOutputTensor(output_names[0]);
    std::vector<int> output_shape = output_t->shape();
    int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
    
    out_data.resize(out_num);
    output_t->copy_to_cpu(out_data.data());

    std::iota(index.begin(), index.end(), 0);
    std::sort(index.begin(), index.end(), [out_data](size_t i1, size_t i2) {
      return out_data[i1] > out_data[i2];
    });
    test_num++;
    if (label == index[0]) {
      top1_num++;
    }
    for (int i = 0; i < 5; i++) {
      if (label == index[i]) {
        top5_num++;
      }
    }
  }
  LOG(INFO) << "final result:";
  LOG(INFO) << "top1 acc:" << top1_num / test_num;
  LOG(INFO) << "top5 acc:" << top5_num / test_num;

  return true;
}
}  // namespace paddle

int main(int argc,char* argv[]) {
     google::ParseCommandLineFlags(&argc, &argv, false);
  for (int i = 0; i < 1; i++) {
    paddle::test_map_cnn(1 << i, 1);
  }
  return 0;
}
