#ifndef __CLUSTER_H__
#define __CLUSTER_H__

#include "../feature_manager.h"

#include <vector>

#include <eigen3/Eigen/Dense>

// TODO(davencyw): move to cpp

class ClusterAlgorithm {
public:
  virtual void cluster(FeatureManager &f_manager) = 0;
  void fillData(FeatureManager &f_manager) {

    for (auto &it_per_id : f_manager.feature) {
      features.push_back(&it_per_id);
      coordinates.push_back(it_per_id.feature_per_frame.back().uv);
      weights.push_back(it_per_id.weight);
    }
  }

  void getClusterHull() {
    // TODO(davencyw): implement
  }

protected:
  std::vector<FeaturePerId *> features;
  std::vector<Vector2d> coordinates;
  std::vector<double> weights;

  std::vector<std::vector>> clusterhulls;
};

class Kmeans : public ClusterAlgorithm {
public:
  void cluster(FeatureManager &f_manager) override {
    fillData(f_manager);
    // TODO(davencyw): implement
    getClusterHull();
  }
};

#endif /* end of include guard: __CLUSTER_H__ */
