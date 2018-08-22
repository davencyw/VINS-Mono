#ifndef __POINTCLASSIFIER_H__
#define __POINTCLASSIFIER_H__

#include "../feature_manager.h"

#include <algorithm>
#include <vector>

class ClassifyPoint {
  bool ready_ = false;
  std::vector<double> maxreprojecterrors;

protected:
  inline double ExponentialWeighting(const double residual,
                                     const double reproject_error_max) {
    double local_weight(0.0);
    double local_residual(0.0);

    local_residual = residual < reproject_error_tolerance_ ? 0 : residual;
    local_residual =
        residual > reproject_error_max ? reproject_error_max : residual;

    local_weight = local_residual / reproject_error_max;

    local_weight = std::pow(expweightdist_, local_weight);

    return 1.0 - (local_weight - 1.0) / (expweightdist_ - 1.0);
  }

  // default values
  double reproject_error_tolerance_ = 0.1;
  double expweightdist_ = 0.1;
  double reproject_error_max_ = 40.0;
  double intermediate_reproject_error_max = 0.0;

  unsigned int num_measurements_ = 100;
  unsigned int current_num_measurements_ = 0;

public:
  bool ready();

  void setParams(const double reproject_error_tolerance,
                 const double expweightdist,
                 const double num_measurements = 100);

  void updateReprojectErrorMax(const double reproject_error_max);

  virtual void classify(FeatureManager &f_manager) = 0;
};

class classifyPointsDep3 : public ClassifyPoint {
public:
  void classify(FeatureManager &f_manager) override;
};

class classifyPointsNoDep : public ClassifyPoint {
public:
  void classify(FeatureManager &f_manager) override;
};

#endif /* end of include guard: __POINTCLASSIFIER_H__ */
