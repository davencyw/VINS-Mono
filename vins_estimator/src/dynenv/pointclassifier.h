#ifndef __POINTCLASSIFIER_H__
#define __POINTCLASSIFIER_H__

#include "../feature_manager.h"
#include <vector>
inline void ExponentialWeighting(const double reproject_error_tolerance,
                                 const double reproject_error_max,
                                 const double expweightdist,
                                 const double residual, double *new_weight) {
  double local_weight(0.0);
  double local_residual(0.0);

  local_residual = residual < reproject_error_tolerance ? 0 : residual;
  local_residual =
      residual > reproject_error_max ? reproject_error_max : residual;

  local_weight = local_residual / reproject_error_max;

  local_weight = std::pow(expweightdist, local_weight);

  *new_weight = 1.0 - (local_weight - 1.0) / (expweightdist - 1.0);
}

class classifyPoint {
protected:
  static constexpr double reproject_error_tolerance = 0.01;
  static constexpr double reproject_error_max = 5.0;
  static constexpr double expweightdist = 0.5;

public:
  virtual void classify(FeatureManager &f_manager) = 0;
};

class classifyPointsDep3 : public classifyPoint {
public:
  void classify(FeatureManager &f_manager) {
    for (auto &it_per_id : f_manager.feature) {
      const double old_weight = it_per_id.weight;
      ExponentialWeighting(reproject_error_tolerance, reproject_error_max,
                           expweightdist, it_per_id.residual,
                           &(it_per_id.weight));

      const double diff = it_per_id.weight - old_weight;
      const double local_new_weight = old_weight + 0.5 * diff;
      it_per_id.weight = local_new_weight;
    }
  }
};

class classifyPointsNoDep : public classifyPoint {
public:
  void classify(FeatureManager &f_manager) override {
    for (auto &it_per_id : f_manager.feature) {
      ExponentialWeighting(reproject_error_tolerance, reproject_error_max,
                           expweightdist, it_per_id.residual,
                           &(it_per_id.weight));
    }
  }
};

#endif /* end of include guard: __POINTCLASSIFIER_H__ */
