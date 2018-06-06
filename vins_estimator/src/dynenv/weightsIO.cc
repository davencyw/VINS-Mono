#include "weightsIO.h"

IO::IO(std::string weights_filepath) {
  _fstream_output_weights = new std::ofstream();
  _fstream_output_weights->open(weights_filepath);
}

IO::~IO() {

  _fstream_output_weights->close();
  delete _fstream_output_weights;
}

void IO::write() {

  for (auto &average_weight_i : _average_weights) {
    *_fstream_output_weights << average_weight_i << ";";
  }

  std::cout << "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n";
  std::cout << "WROTE WEIGHTS!" << std::endl;
}

void IO::averageWeights2File(const FeatureManager &f_manager) {

  double average_weight(0.0);
  for (auto &it_per_id : f_manager.feature) {
    average_weight += it_per_id.weight;
  }

  average_weight /= static_cast<double>(f_manager.feature.size());
  std::cout << average_weight;
  _average_weights.push_back(average_weight);
}
