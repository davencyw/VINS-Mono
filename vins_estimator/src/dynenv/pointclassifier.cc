#include "pointclassifier.h"

bool ClassifyPoint::ready() { return ready_; }

void ClassifyPoint::setParams(const double reproject_error_tolerance,
                              const double expweightdist,
                              const double num_measurements) {
  reproject_error_tolerance_ = reproject_error_tolerance;
  expweightdist_ = expweightdist;
  num_measurements_ = num_measurements;
}

void ClassifyPoint::updateReprojectErrorMax(const double reproject_error_max) {

  // skip outliers in initalization
  if (reproject_error_max > 420) {
    return;
  }

  intermediate_reproject_error_max += reproject_error_max;
  maxreprojecterrors.push_back(reproject_error_max);

  if (++current_num_measurements_ > num_measurements_) {
    // median
    std::nth_element(maxreprojecterrors.begin(),
                     maxreprojecterrors.begin() + maxreprojecterrors.size() / 2,
                     maxreprojecterrors.end());
    reproject_error_max_ =
        maxreprojecterrors[maxreprojecterrors.size() / 2] * 3.0;

    std::cout << "\n found max reprojecterror: " << reproject_error_max_
              << "\n";
    ready_ = true;
  }
}

void classifyPointsDep3::classify(FeatureManager &f_manager) {
  for (auto &it_per_id : f_manager.feature) {
    const double local_new_weight =
        0.5 * it_per_id.weight +
        0.5 * ExponentialWeighting(it_per_id.residual, reproject_error_max_);
    it_per_id.weight = local_new_weight;
  }
}

void classifyPointsNoDep::classify(FeatureManager &f_manager) {
  for (auto &it_per_id : f_manager.feature) {
    it_per_id.weight =
        ExponentialWeighting(it_per_id.residual, reproject_error_max_);
  }
}
