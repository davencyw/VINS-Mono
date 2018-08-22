#ifndef __CLUSTER_H__
#define __CLUSTER_H__

#include "../feature_manager.h"

#include <algorithm>
#include <vector>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <eigen3/Eigen/Dense>

// TODO(davencyw): move to cpp

struct Cluster {
  Vector2d center;
  std::vector<cv::Point> convexhull;
  Vector2d averageopticalflow;
  double averageweight;
};

class ClusterAlgorithm {
public:
  virtual void cluster(FeatureManager &f_manager, const int framecount,
                       std::deque<Cluster> &cluster) = 0;

protected:
  static constexpr double clusterthreshold_ = 0.3;
};

// implementation for only one single cluster!
class SimpleCluster : public ClusterAlgorithm {

private:
  const unsigned _cluster_windowsize;
  const unsigned _num_cluster_confirmation;

  // prior for cluster in
  std::vector<cv::Point> _convexclusterhull;

public:
  SimpleCluster(const unsigned cluster_windowsize,
                const unsigned num_cluster_confirmation)
      : _cluster_windowsize(cluster_windowsize),
        _num_cluster_confirmation(num_cluster_confirmation){};

  void cluster(FeatureManager &f_manager, const int framecount,
               std::deque<Cluster> &cluster) override;
};

#endif /* end of include guard: __CLUSTER_H__ */
