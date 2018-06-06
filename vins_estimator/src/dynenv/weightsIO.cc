#include "weightsIO.h"

IO::IO(std::string weights_filepath) : _weights_filepath(weights_filepath) {}

void IO::write() {
  _weights_file.open(_weights_filepath, ios::out);

  for (auto &average_weight_i : _average_weights) {
    _weights_file << average_weight_i << ";";
  }
  _weights_file.close();
  std::cout << "\nWROTE " << _average_weights.size() << " WEIGHTS!"
            << std::endl;
}

void IO::averageWeights2File(const FeatureManager &f_manager) {

  double average_weight(0.0);
  for (auto &it_per_id : f_manager.feature) {
    average_weight += it_per_id.weight;
  }

  average_weight /= static_cast<double>(f_manager.feature.size());
  _average_weights.push_back(average_weight);
}
