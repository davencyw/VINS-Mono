#ifndef __WEIGHTSIO_H__
#define __WEIGHTSIO_H__

#include "../feature_manager.h"

#include <fstream>
#include <string>

class IO {
public:
  IO(std::string weights_filepath);
  void weights2File(const FeatureManager &f_manager);

private:
  std::string weights_filepath;
  std::unique_ptr<std::ofstream> _fstream_output_weights;
};

#endif /* end of include guard: __WEIGHTSIO_H__ */
