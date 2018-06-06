#include "weightsIO.h"

IO::IO(std::string weights_filepath) {
  _fstream_output_weights = std::make_unique<std::ofstream>();
  _fstream_output_weights->open(weights_filepath);
}
void IO::weights2File(const FeatureManager &f_manager) {}
