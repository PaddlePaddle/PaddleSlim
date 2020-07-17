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
#ifdef WITH_GPERFTOOLS
#include <gperftools/profiler.h>
#include <paddle/fluid/platform/profiler.h>
#endif

DEFINE_string(infer_model, "", "path to the model");
DEFINE_string(infer_data, "", "path to the input data");
DEFINE_int32(batch_size, 50, "inference batch size");
DEFINE_int32(iterations,
             0,
             "number of batches to process. 0 means testing whole dataset");
DEFINE_int32(num_threads, 1, "num of threads to run in parallel");
DEFINE_bool(with_accuracy_layer,
            true,
            "Set with_accuracy_layer to true if provided model has accuracy "
            "layer and requires label input");
DEFINE_bool(use_profile,
            false,
            "Set use_profile to true to get profile information");
DEFINE_bool(use_analysis,
            false,
            "If use_analysis is set to true, the model will be optimized");

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
    if (i > 0 && i % 100) {
      LOG(INFO) << "Read " << i * 100 * FLAGS_batch_size << " samples";
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

void PredictionRun(paddle::PaddlePredictor *predictor,
                   const std::vector<std::vector<paddle::PaddleTensor>> &inputs,
                   std::vector<std::vector<paddle::PaddleTensor>> *outputs,
                   int num_threads,
                   float *sample_latency = nullptr) {
  int iterations = inputs.size();  // process the whole dataset ...
  if (FLAGS_iterations > 0 &&
      FLAGS_iterations < static_cast<int64_t>(inputs.size()))
    iterations =
        FLAGS_iterations;  // ... unless the number of iterations is set
  outputs->resize(iterations);
  Timer run_timer;
  double elapsed_time = 0;
#ifdef WITH_GPERFTOOLS
  ResetProfiler();
  ProfilerStart("paddle_inference.prof");
#endif
  int predicted_num = 0;

  for (int i = 0; i < iterations; i++) {
    run_timer.tic();
    predictor->Run(inputs[i], &(*outputs)[i], FLAGS_batch_size);
    elapsed_time += run_timer.toc();

    predicted_num += FLAGS_batch_size;
    if (predicted_num % 100 == 0) {
      LOG(INFO) << "Infer " << predicted_num << " samples";
    }
  }

#ifdef WITH_GPERFTOOLS
  ProfilerStop();
#endif

  auto batch_latency = elapsed_time / iterations;
  PrintTime(FLAGS_batch_size, num_threads, batch_latency, iterations);

  if (sample_latency != nullptr)
    *sample_latency = batch_latency / FLAGS_batch_size;
}

std::pair<float, float> CalculateAccuracy(
    const std::vector<std::vector<paddle::PaddleTensor>> &outputs,
    const std::vector<paddle::PaddleTensor> &labels_gt,
    bool with_accuracy = FLAGS_with_accuracy_layer) {
  LOG_IF(ERROR, !with_accuracy && labels_gt.size() == 0)
      << "if with_accuracy set to false, labels_gt must be not empty";
  std::vector<float> acc1_ss;
  std::vector<float> acc5_ss;
  if (!with_accuracy) {     // model with_accuracy_layer = false
    float *result_array;    // for one batch 50*1000
    int64_t *batch_labels;  // 50*1
    LOG_IF(ERROR, outputs.size() != labels_gt.size())
        << "outputs first dimension must be equal to labels_gt first dimension";
    for (auto i = 0; i < outputs.size();
         ++i) {  // same as labels first dimension
      result_array = static_cast<float *>(outputs[i][0].data.data());
      batch_labels = static_cast<int64_t *>(labels_gt[i].data.data());
      int correct_1 = 0, correct_5 = 0, total = FLAGS_batch_size;
      for (auto j = 0; j < FLAGS_batch_size; j++) {  // batch_size
        std::vector<float> v(result_array + j * 1000,
                             result_array + (j + 1) * 1000);
        std::vector<std::pair<float, int>> vx;
        for (int k = 0; k < 1000; k++) {
          vx.push_back(std::make_pair(v[k], k));
        }
        std::partial_sort(vx.begin(),
                          vx.begin() + 5,
                          vx.end(),
                          [](std::pair<float, int> a, std::pair<float, int> b) {
                            return a.first > b.first;
                          });
        if (static_cast<int>(batch_labels[j]) == vx[0].second) correct_1 += 1;
        if (std::find_if(vx.begin(),
                         vx.begin() + 5,
                         [batch_labels, j](std::pair<float, int> a) {
                           return static_cast<int>(batch_labels[j]) == a.second;
                         }) != vx.begin() + 5)
          correct_5 += 1;
      }
      acc1_ss.push_back(static_cast<float>(correct_1) /
                        static_cast<float>(total));
      acc5_ss.push_back(static_cast<float>(correct_5) /
                        static_cast<float>(total));
    }
  } else {  // model with_accuracy_layer = true
    for (auto i = 0; i < outputs.size(); ++i) {
      LOG_IF(ERROR, outputs[i].size() < 3UL) << "To get top1 and top5 "
                                                "accuracy, output[i] size must "
                                                "be bigger than or equal to 3";
      acc1_ss.push_back(
          *static_cast<float *>(outputs[i][1].data.data()));  // 1 is top1 acc
      acc5_ss.push_back(*static_cast<float *>(
          outputs[i][2].data.data()));  // 2 is top5 acc or mAP
    }
  }
  auto acc1_ss_avg =
      std::accumulate(acc1_ss.begin(), acc1_ss.end(), 0.0) / acc1_ss.size();
  auto acc5_ss_avg =
      std::accumulate(acc5_ss.begin(), acc5_ss.end(), 0.0) / acc5_ss.size();
  return std::make_pair(acc1_ss_avg, acc5_ss_avg);
}

static void SetIrOptimConfig(paddle::AnalysisConfig *cfg) {
  cfg->DisableGpu();
  cfg->SwitchIrOptim();
  cfg->EnableMKLDNN();
  if (FLAGS_use_profile) {
    cfg->EnableProfile();
  }
}

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

  std::vector<std::vector<paddle::PaddleTensor>> input_slots_all;
  std::vector<std::vector<paddle::PaddleTensor>> outputs;
  std::vector<paddle::PaddleTensor> labels_gt;  // optional
  SetInput(&input_slots_all, &labels_gt);       // iterations*batch_size
  auto predictor =
      CreatePredictor(reinterpret_cast<paddle::PaddlePredictor::Config *>(&cfg),
                      FLAGS_use_analysis);
  PredictionRun(predictor.get(), input_slots_all, &outputs, FLAGS_num_threads);
  auto acc_pair = CalculateAccuracy(outputs, labels_gt);
  LOG(INFO) << "Top1 accuracy: " << std::fixed << std::setw(6)
            << std::setprecision(4) << acc_pair.first;
  LOG(INFO) << "Top5 accuracy: " << std::fixed << std::setw(6)
            << std::setprecision(4) << acc_pair.second;
}
