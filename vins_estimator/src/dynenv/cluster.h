#ifndef __CLUSTER_H__
#define __CLUSTER_H__

#include "../feature_manager.h"

#include <algorithm>
#include <vector>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <eigen3/Eigen/Dense>

struct Cluster {
  Vector2d center;
  std::vector<cv::Point> convexhull;
  Vector2d averageopticalflow;
  double averageweight;
  int clusterid;
};

class ClusterAlgorithm {
public:
  void cluster(FeatureManager &f_manager, const int framecount,
               std::deque<std::vector<Cluster>> &cluster);

protected:
  ClusterAlgorithm(const unsigned cluster_windowsize,
                   const unsigned num_cluster_confirmation)
      : _cluster_windowsize(cluster_windowsize),
        _num_cluster_confirmation(num_cluster_confirmation){};

  void selectPointsAndReduce(
      FeatureManager &f_manager,
      std::vector<std::pair<FeaturePerId *, double>> &cluster_candidates,
      std::deque<std::vector<Cluster>> &cluster, const int framecount);
  void updatePoints();
  void moveCluster(std::deque<std::vector<Cluster>> &cluster);
  void addInliers(FeatureManager &f_manager,
                  const std::vector<Cluster> &new_cluster);

  virtual std::vector<Cluster> computecluster(
      std::vector<std::pair<FeaturePerId *, double>> &cluster_candidates) = 0;

  static constexpr double clusterthreshold_ = 0.3;
  const unsigned _cluster_windowsize;
  const unsigned _num_cluster_confirmation;
};

// implementation for only one single cluster!
class SimpleCluster : public ClusterAlgorithm {
public:
  SimpleCluster(const unsigned cluster_windowsize,
                const unsigned num_cluster_confirmation)
      : ClusterAlgorithm(cluster_windowsize, num_cluster_confirmation){};

  std::vector<Cluster> computecluster(
      std::vector<std::pair<FeaturePerId *, double>> &cluster_candidates)
      override;
};

// multi object/cluster support
class DbscanCluster : public ClusterAlgorithm {
public:
  DbscanCluster(const unsigned cluster_windowsize,
                const unsigned num_cluster_confirmation)
      : ClusterAlgorithm(cluster_windowsize, num_cluster_confirmation){};

  std::vector<Cluster> computecluster(
      std::vector<std::pair<FeaturePerId *, double>> &cluster_candidates)
      override;

private:
  void
  dbscan(std::vector<std::pair<FeaturePerId *, double>> &cluster_candidates,
         const double eps, const unsigned int minpts);
};

#endif /* end of include guard: __CLUSTER_H__ */
