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
#include <thread>
#include <iostream>

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
DEFINE_int32(num_jobs, 2, "num of threads to run in parallel");

std::unique_ptr<paddle::PaddlePredictor> CreatePredictor(
    const paddle::PaddlePredictor::Config *config, bool use_analysis = true) {
  const auto *analysis_config =
      reinterpret_cast<const paddle::AnalysisConfig *>(config);
  if (use_analysis) {
    return paddle::CreatePaddlePredictor<paddle::AnalysisConfig>(
        *analysis_config);
  }
  auto native_config = analysis_config->ToNativeConfig();
  return paddle::CreatePaddlePredictor<paddle::NativeConfig>(native_config);
}

static void SetIrOptimConfig(paddle::AnalysisConfig *cfg) {
  cfg->DisableGpu();
  cfg->SwitchIrOptim();
  cfg->EnableMKLDNN();
  cfg->SetModel(FLAGS_infer_model);
  cfg->SetCpuMathLibraryNumThreads(FLAGS_num_threads);
}

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

using Time = decltype(std::chrono::high_resolution_clock::now());
Time time() { return std::chrono::high_resolution_clock::now(); };
double time_diff(Time t1, Time t2) {
  typedef std::chrono::microseconds ms;
  auto diff = t2 - t1;
  ms counter = std::chrono::duration_cast<ms>(diff);
  return counter.count() / 1000.0;
}

template <typename T>
constexpr paddle::PaddleDType GetPaddleDType();

template <>
constexpr paddle::PaddleDType GetPaddleDType<int64_t>() {
  return paddle::PaddleDType::INT64;
}

template <>
constexpr paddle::PaddleDType GetPaddleDType<float>() {
  return paddle::PaddleDType::FLOAT32;
}

template <typename T>
class TensorReader {
 public:
  TensorReader(std::ifstream &file,
               size_t beginning_offset,
               std::vector<int> shape,
               std::string name)
      : file_(file), position_(beginning_offset), shape_(shape), name_(name) {
    numel_ = std::accumulate(
        shape_.begin(), shape_.end(), size_t{1}, std::multiplies<size_t>());
  }

  paddle::PaddleTensor NextBatch() {
    paddle::PaddleTensor tensor;
    tensor.name = name_;
    tensor.shape = shape_;
    tensor.dtype = GetPaddleDType<T>();
    tensor.data.Resize(numel_ * sizeof(T));
    file_.seekg(position_);
    file_.read(static_cast<char *>(tensor.data.data()), numel_ * sizeof(T));
    position_ = file_.tellg();
    if (file_.eof()) LOG(ERROR) << name_ << ": reached end of stream";
    if (file_.bad()) LOG(ERROR) << name_ << "ERROR: badbit is true";
    if (file_.fail())
      throw std::runtime_error(name_ + ": failed reading file.");
    return tensor;
  }

 protected:
  std::ifstream &file_;
  size_t position_;
  std::vector<int> shape_;
  std::string name_;
  size_t numel_;
};

void SetInput(std::vector<std::vector<paddle::PaddleTensor>> *inputs,
              std::vector<paddle::PaddleTensor> *labels_gt,
              bool with_accuracy_layer = FLAGS_with_accuracy_layer,
              int32_t batch_size = FLAGS_batch_size) {
  std::ifstream file(FLAGS_infer_data, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Couldn't open file: " + FLAGS_infer_data);
  }

  int64_t total_images{0};
  file.seekg(0, std::ios::beg);
  file.read(reinterpret_cast<char *>(&total_images), sizeof(total_images));
  LOG(INFO) << "Total images in file: " << total_images;

  std::vector<int> image_batch_shape{batch_size, 3, 224, 224};
  std::vector<int> label_batch_shape{batch_size, 1};
  auto images_offset_in_file = static_cast<size_t>(file.tellg());

  TensorReader<float> image_reader(
      file, images_offset_in_file, image_batch_shape, "image");

  auto iterations_max = total_images / batch_size;
  auto iterations = iterations_max;
  if (FLAGS_iterations > 0 && FLAGS_iterations < iterations_max) {
    iterations = FLAGS_iterations;
  }

  auto labels_offset_in_file =
      images_offset_in_file + sizeof(float) * total_images * 3 * 224 * 224;

  TensorReader<int64_t> label_reader(
      file, labels_offset_in_file, label_batch_shape, "label");
  for (auto i = 0; i < iterations; i++) {
    auto images = image_reader.NextBatch();
    std::vector<paddle::PaddleTensor> tmp_vec;
    tmp_vec.push_back(std::move(images));
    auto labels = label_reader.NextBatch();
    if (with_accuracy_layer) {
      tmp_vec.push_back(std::move(labels));
    } else {
      labels_gt->push_back(std::move(labels));
    }
    inputs->push_back(std::move(tmp_vec));
    if (i > 0 && i % 100==0)  {
      LOG(INFO) << "Read " << i * FLAGS_batch_size << " samples";
    }
  }
}

static void PrintTime(int batch_size,
                      int num_threads,
                      double batch_latency,
                      int epoch = 1) {
  double sample_latency = batch_latency / batch_size;
  LOG(INFO) << "Model: " << FLAGS_infer_model;
  LOG(INFO) << "====== num of threads: " << num_threads << " ======";
  LOG(INFO) << "====== batch size: " << batch_size << ", iterations: " << epoch;
  LOG(INFO) << "====== batch latency: " << batch_latency
            << "ms, number of samples: " << batch_size * epoch;
  LOG(INFO) << ", sample latency: " << sample_latency
            << "ms, fps: " << 1000.f / sample_latency << " ======";
}

bool PredictionRun(int num_jobs,
                   const std::vector<std::vector<paddle::PaddleTensor>> &inputs,
                   int num_threads,
                   float *sample_latency = nullptr) {
  std::cout<<"LOG INFO 1-----------------------------";
  
  std::cout<<"LOG INFO 2.1 2.1 2.1 2.1 2.1 2.1 2.1 2.1 2.1 2.1 -----------------------------";
  
  paddle::AnalysisConfig cfg;
  paddle::AnalysisConfig cfg1;

  SetIrOptimConfig(&cfg);
  SetIrOptimConfig(&cfg1);
  std::cout<<"LOG INFO 2.51 2.51 2.51 2.51-----------------------------------------------------------------";
 
  paddle::PaddlePredictor *pres[2]; 
  auto predictor0 = CreatePredictor(reinterpret_cast<paddle::PaddlePredictor::Config *>(&cfg));
  auto predictor1 = CreatePredictor(reinterpret_cast<paddle::PaddlePredictor::Config *>(&cfg1));

  std::cout<<"LOG INFO 2.52 2.52 2.52 2.52-----------------------------------------------------";
  
  pres[0] = predictor0.get();
  pres[1] = predictor1.get();

  std::cout<<"LOG INFO 2.5 2.5 2.5 2.5-----------------------------";
  
  std::vector<std::thread> threads;
  std::cout<<"LOG INFO 3333333333-----------------------------";
  auto time1 = time();
  
  num_jobs=2;
  for (int tid = 0; tid< num_jobs; tid++){
    threads.emplace_back([&, tid](){
      std::vector<std::vector<paddle::PaddleTensor>> *outputs;
      for (int j = 0 ; j < FLAGS_iterations;j++){
        outputs->resize(FLAGS_iterations);
        pres[tid]->Run(inputs[j], &(*outputs)[j], FLAGS_batch_size);
      }
    });
  }

  for(int i = 0; i < num_jobs; i++){
    threads[i].join();
  }
  // auto time2 = time(); 
  // std::cout << " predict cost: " << time_diff(time1, time2) / 4000.0 << "ms" << std::endl;

  return true;
}

int main(int argc, char *argv[]) {
  // InitFLAGS(argc, argv);
  google::InitGoogleLogging(*argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::vector<std::vector<paddle::PaddleTensor>> input_slots_all;
  std::vector<paddle::PaddleTensor> labels_gt;  // optional
  SetInput(&input_slots_all, &labels_gt);       // iterations*batch_size
  
  PredictionRun(FLAGS_num_jobs, input_slots_all, FLAGS_num_threads);
  // auto acc_pair = CalculateAccuracy(outputs, labels_gt);
  // LOG(INFO) << "Top1 accuracy: " << std::fixed << std::setw(6)
  //           << std::setprecision(4) << acc_pair.first;
  // LOG(INFO) << "Top5 accuracy: " << std::fixed << std::setw(6)
  //           << std::setprecision(4) << acc_pair.second;
  return 0;
}
