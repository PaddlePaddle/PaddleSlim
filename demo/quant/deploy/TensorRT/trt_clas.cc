#include <numeric>
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <chrono>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "paddle/include/paddle_inference_api.h"

DEFINE_string(model_dir, "", "Directory of the inference model.");
DEFINE_int32(batch_size, 1, "Batch size.");
DEFINE_bool(use_calib, true, "Whether to use calib. Set to true if you are using TRT calibration; \
                             Set to false if you are using PaddleSlim quant models.");
DEFINE_string(data_file, "", "Path of the inference data file.");
DEFINE_bool(use_int8, true, "use trt int8 or not");
DEFINE_int32(repeat_times, 1000, "benchmark repeat time");

using Time = decltype(std::chrono::high_resolution_clock::now());
Time time() { return std::chrono::high_resolution_clock::now(); };
double time_diff(Time t1, Time t2) {
  typedef std::chrono::microseconds ms;
  auto diff = t2 - t1;
  ms counter = std::chrono::duration_cast<ms>(diff);
  return counter.count() / 1000.0;
}

std::pair<int, float> findMax(std::vector<float>& vec) {
	float max = -1000;
	int idx = -1;
	for (int i=0; i<vec.size(); ++i) {
		if (vec[i] > max) {
			max = vec[i];
			idx=i;
		}
	}
	return {idx, max};
}

std::unique_ptr<paddle::PaddlePredictor> CreatePredictor() {
  paddle::AnalysisConfig config;
  config.SetModel(FLAGS_model_dir + "/model", FLAGS_model_dir + "/params");
  config.EnableUseGpu(500, 0);
  // We use ZeroCopy, so we set config.SwitchUseFeedFetchOps(false) here.
  config.SwitchUseFeedFetchOps(false);
  if (FLAGS_use_int8){
    config.EnableTensorRtEngine(1 << 30, \
                                FLAGS_batch_size, \
                                5, \
                                paddle::AnalysisConfig::Precision::kInt8, \
                                false, \
                                FLAGS_use_calib);
  }
  else{
    config.EnableTensorRtEngine(1 << 30, \
                                FLAGS_batch_size, \
                                5, \
                                paddle::AnalysisConfig::Precision::kFloat32, \
                                false, \
                                false);
  }
  return CreatePaddlePredictor(config);
}

void run(paddle::PaddlePredictor *predictor,
         const std::vector<float>& input,
         const std::vector<int>& input_shape, 
         std::vector<float> *out_data) {
  int input_num = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<int>());

  auto input_names = predictor->GetInputNames();
  auto input_t = predictor->GetInputTensor(input_names[0]);
  input_t->Reshape(input_shape);
  input_t->copy_from_cpu(input.data());

  // warm up 10 times
  if (FLAGS_repeat_times != 1) {
      for(int i=0; i<10; i++){
        CHECK(predictor->ZeroCopyRun());
      }
      LOG(INFO) << "finish warm up 10 times";
  }

  auto time1 = time(); 
  for(int i=0; i<FLAGS_repeat_times; i++){
    CHECK(predictor->ZeroCopyRun());
    auto output_names = predictor->GetOutputNames();
    // there is only one output of Resnet50
    auto output_t = predictor->GetOutputTensor(output_names[0]);
    std::vector<int> output_shape = output_t->shape();
    int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());

    out_data->resize(out_num);
    output_t->copy_to_cpu(out_data->data());
  }
  auto time2 = time(); 
  auto total_time = time_diff(time1, time2);
  auto average_time = total_time / FLAGS_repeat_times;
  LOG(INFO) << "total predict cost is : " << total_time << " ms, repeat " << FLAGS_repeat_times << " times";
  LOG(INFO) << "average predict cost is : " << average_time << " ms";
  LOG(INFO) << "finish prediction";
}
 
int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  auto predictor = CreatePredictor();
  std::vector<int> input_shape = {FLAGS_batch_size, 3, 224, 224}; 
  // Init input as 1.0 here for example. You can also load preprocessed real pictures to vectors as input.
  // std::vector<float> input_vec(FLAGS_batch_size * 3 * 224 * 224, 1.0);

  int input_num = FLAGS_batch_size * 3 * 224 * 224;
  float *input_data = new float[input_num];
  memset(input_data, 0, input_num * sizeof(float));

  std::ifstream fs(FLAGS_data_file, std::ifstream::binary);
  if (!fs.is_open()) {
    LOG(FATAL) << "open input file fail.";
  }
  auto input_data_tmp = input_data;
  for (int i = 0; i < input_num; ++i) {
      fs.read(reinterpret_cast<char*>(input_data_tmp), sizeof(*input_data_tmp));
      input_data_tmp++;
  }
  int label = 0;
  fs.read(reinterpret_cast<char*>(&label), sizeof(label));
  fs.close();

  std::vector<float> input_vec {input_data, input_data + input_num};

  std::vector<float> out_data;
  run(predictor.get(), input_vec, input_shape, &out_data);
  if (FLAGS_batch_size == 1) {
      std::pair<int, float> result = findMax(out_data);
      LOG(INFO) << "pred image class is : " << result.first << ", ground truth label is : " << label;
  }
  return 0;
}

