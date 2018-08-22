#include "cluster.h"

void SimpleCluster::cluster(FeatureManager &f_manager, const int framecount,
                            std::deque<Cluster> &cluster) {

  // move all clusters to next frame to use as prior
  // TODO(dave): add time dependency on velocity
  bool cluster_available = !cluster.empty();
  if (cluster_available) {
    // adjust all clusters with flow from last frame
    const double averageopticalflow_x(cluster.back().averageopticalflow.x());
    const double averageopticalflow_y(cluster.back().averageopticalflow.y());
    for (auto &cluster_i : cluster) {
      for (auto &point_i : cluster_i.convexhull) {
        point_i.x += averageopticalflow_x * 2.0;
        point_i.y += averageopticalflow_y * 2.0;
      }
    }
  }

  // cluster center for cluster-points with low weights
  int num_in_cluster(0);
  Vector2d center(0, 0);
  std::vector<std::pair<FeaturePerId *, double>> features_in_cluster;

  for (auto &it_per_id : f_manager.feature) {

    it_per_id.clusterid = 0;

    if (it_per_id.start_frame + it_per_id.feature_per_frame.size() <
        framecount) {
      continue;
    }

    // check if feature is inside a moved cluster from old frames
    if (cluster_available) {
      const auto puv(it_per_id.feature_per_frame.back().uv);
      const cv::Point p(puv.x(), puv.y());
      const int numclusters(cluster.size());
      int innumclusters(0);

      for (auto &cluster_i : cluster) {
        if (!cluster_i.convexhull.empty()) {
          const double result =
              cv::pointPolygonTest(cluster_i.convexhull, p, false);
          if (result > -1) {
            ++innumclusters;
          }
        }
      }
      if (innumclusters) {
        // inside polygon of prior moved cluster
        double percentageofclusters(
            static_cast<double>(innumclusters) /
            static_cast<double>(_num_cluster_confirmation));
        percentageofclusters > 1.0
            ? percentageofclusters = 1.0
            : percentageofclusters = percentageofclusters;

        constexpr double multiplier(clusterthreshold_ - 0.01);
        constexpr double diff(1 - multiplier);
        it_per_id.weight *= (1 - diff * percentageofclusters);
      }
    }

    if (it_per_id.weight < clusterthreshold_) {
      it_per_id.clusterid = 1;
      center += it_per_id.feature_per_frame.back().uv;
      features_in_cluster.push_back(std::make_pair(&it_per_id, 0));
    }
  }

  Vector2d averageopticalflow(0.0, 0.0);
  double averageweight(0.0);

  if (features_in_cluster.size()) {
    center /= static_cast<double>(features_in_cluster.size());
    double averagedist(0.0);
    // compute distanecs to center
    for (auto &it_per_id : features_in_cluster) {
      const Vector2d dist(it_per_id.first->feature_per_frame.back().uv -
                          center);
      const double distscalar(dist.norm());
      averagedist += distscalar;
      it_per_id.second = distscalar;
    }

    averagedist /= static_cast<double>(features_in_cluster.size());

    // remove outliers
    features_in_cluster.erase(
        std::remove_if(features_in_cluster.begin(), features_in_cluster.end(),
                       [=](std::pair<FeaturePerId *, double> i) {
                         // remove from cluster if distance is bigger than 2
                         // times the averagedistance
                         bool outlier(i.second > 2.0 * averagedist);
                         if (outlier) {
                           i.first->clusterid = -1;
                         }
                         return outlier;
                       }),
        features_in_cluster.end());

    // compute new clustercenter and averageweight*
    center = Vector2d(0, 0);

    std::vector<cv::Point> inputarray;
    for (auto &it_per_id : features_in_cluster) {
      const auto puv = it_per_id.first->feature_per_frame.back().uv;
      inputarray.push_back(cv::Point(puv.x(), puv.y()));
      center += puv;
      averageweight += it_per_id.first->weight;
      averageopticalflow += it_per_id.first->feature_per_frame.back().velocity;
    }
    averageweight /= static_cast<double>(features_in_cluster.size());
    center /= static_cast<double>(features_in_cluster.size());

    // add inliers
    cv::convexHull(inputarray, _convexclusterhull);

    for (auto &it_per_id : f_manager.feature) {
      if (it_per_id.clusterid == 0) {

        const auto puv(it_per_id.feature_per_frame.back().uv);
        const cv::Point p(puv.x(), puv.y());
        const double result =
            cv::pointPolygonTest(_convexclusterhull, p, false);

        if (result > -1) {
          // inside polygon
          it_per_id.clusterid = 2;
          features_in_cluster.push_back(std::make_pair(&it_per_id, 0));
          it_per_id.weight = averageweight;
        }
      }
    }
  }
  Cluster temp_cluster;
  temp_cluster.center = center;
  temp_cluster.convexhull = _convexclusterhull;
  temp_cluster.averageweight = averageweight;
  temp_cluster.averageopticalflow = averageopticalflow;
  cluster.push_back(temp_cluster);
  if (cluster.size() > _cluster_windowsize) {
    cluster.pop_front();
  }
};
