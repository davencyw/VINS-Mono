#ifndef __WEIGHTSIO_H__
#define __WEIGHTSIO_H__

#include "../feature_manager.h"

#include <fstream>
#include <memory>
#include <string>
#include <vector>

class IO {
public:
  IO(std::string weights_filepath);
  void averageWeights2File(const FeatureManager &f_manager);
  void writeaverage();
  void writestep(const FeatureManager &f_manager);

private:
  std::string _weights_filepath;
  std::string _weights_step_filepath;
  std::ofstream _weights_file;
  std::ofstream _weights_step_file;
  std::vector<double> _average_weights;
};

#endif /* end of include guard: __WEIGHTSIO_H__ */
