#include <vector>

inline void ExponentialWeighting(const double reproject_error_tolerance,
                                 const double reproject_error_max,
                                 const double expweightdist,
                                 std::vector<double> &new_weights) {

  for (size_t i(0); i < new_weights.size(); ++i) {
    new_weights[i] =
        new_weights[i] < reproject_error_tolerance ? 0 : new_weights[i];
    new_weights[i] = new_weights[i] > reproject_error_max ? reproject_error_max
                                                          : new_weights[i];
  }

  for (auto &new_weights_i : new_weights) {
    new_weights_i /= reproject_error_max;
  }

  for (size_t i(0); i < new_weights.size(); ++i) {
    new_weights[i] = std::pow(expweightdist, new_weights[i]);
  }

  for (size_t i(0); i < new_weights.size(); ++i) {
    new_weights[i] =
        new_weights[i] < reproject_error_tolerance ? 0 : new_weights[i];
  }

  for (auto &new_weights_i : new_weights) {
    new_weights_i = 1.0 - (new_weights_i - 1.0) / (expweightdist - 1.0);
  }
}

inline void classifyPointsNoDep(const std::vector<double> &residuals,
                                std::vector<double> &weights) {

  std::vector<double> new_weights = residuals;

  // set tolerance
  constexpr double reproject_error_tolerance(0.01);
  constexpr double reproject_error_max(5.0);
  constexpr double expweightdist(0.5);
  ExponentialWeighting(reproject_error_tolerance, reproject_error_max,
                       expweightdist, new_weights);

  weights = new_weights;
  ;
}
