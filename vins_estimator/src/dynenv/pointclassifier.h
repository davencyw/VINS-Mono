#ifndef __POINTCLASSIFIER_H__
#define __POINTCLASSIFIER_H__

#include "../feature_manager.h"
#include <vector>

class ClassifyPoint {
  bool ready_ = false;

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

  double reproject_error_tolerance_ = 0.1;
  double expweightdist_ = 0.1;
  double reproject_error_max_ = 40.0;
  double intermediate_reproject_error_max = 0.0;

  unsigned int num_measurements_ = 100;
  unsigned int current_num_measurements_ = 0;

public:
  bool ready() { return ready_; }

  void setParams(const double reproject_error_tolerance,
                 const double reproject_error_max, const double expweightdist,
                 const double num_measurements = 100) {
    reproject_error_tolerance_ = reproject_error_tolerance;
    reproject_error_max_ = reproject_error_max;
    expweightdist_ = expweightdist;
    num_measurements_ = num_measurements;
  }

  void updateReprojectErrorMax(const double reproject_error_max) {

    intermediate_reproject_error_max += reproject_error_max;

    if (++current_num_measurements_ > num_measurements_) {
      reproject_error_max_ = (intermediate_reproject_error_max /
                              static_cast<double>(current_num_measurements_)) *
                             3.0;
      std::cout << "\n found max reprojecterror: " << reproject_error_max_
                << "\n";
      ready_ = true;
    }
  }

  virtual void classify(FeatureManager &f_manager) = 0;
};

// TODO(davencyw): move to cpp

class classifyPointsDepthDep3 : public ClassifyPoint {
public:
  void classify(FeatureManager &f_manager) {
    // TODO(davencyw): make zmax dependent on linear and angular imu velocity
    const double zmax(10);

    for (auto &it_per_id : f_manager.feature) {
      // scale reproject_error_max with z
      // TODO(davencyw): change reproject_error_max maximum (400) to adapt with
      // image size
      double z(it_per_id.estimated_depth);
      z == -1 ? z = zmax : z = z;
      const double local_reproject_error_max(
          (z - 1) * (200 - reproject_error_max_) / (zmax - 1) +
          reproject_error_max_);

      const double local_new_weight =
          0.5 * it_per_id.weight +
          0.5 * ExponentialWeighting(it_per_id.residual,
                                     local_reproject_error_max);
      it_per_id.weight = local_new_weight;
    }
  }
};

class classifyPointsDep3 : public ClassifyPoint {
public:
  void classify(FeatureManager &f_manager) {
    for (auto &it_per_id : f_manager.feature) {
      const double local_new_weight =
          0.5 * it_per_id.weight +
          0.5 * ExponentialWeighting(it_per_id.residual, reproject_error_max_);
      it_per_id.weight = local_new_weight;
    }
  }
};

class classifyPointsNoDep : public ClassifyPoint {
public:
  void classify(FeatureManager &f_manager) override {
    for (auto &it_per_id : f_manager.feature) {
      it_per_id.weight =
          ExponentialWeighting(it_per_id.residual, reproject_error_max_);
    }
  }
};

#endif /* end of include guard: __POINTCLASSIFIER_H__ */
