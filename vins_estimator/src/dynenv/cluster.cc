#include "cluster.h"

void ClusterAlgorithm::cluster(FeatureManager &f_manager, const int framecount,
                               std::deque<std::vector<Cluster>> &cluster) {

  std::vector<std::pair<FeaturePerId *, double>> cluster_candidates;

  // TODO(davencyw):
  // move clusters
  moveCluster(cluster);
  // find and reduce cluster-inlier, select points
  selectPoints(f_manager, cluster_candidates, cluster, framecount);
  // cluster points
  std::vector<Cluster> new_cluster(
      computecluster(f_manager, cluster_candidates, framecount));
  addInliers(f_manager, new_cluster);
  // update weight

  if (cluster.size() > _cluster_windowsize) {
    cluster.pop_front();
  }
}

void ClusterAlgorithm::selectPoints(
    FeatureManager &f_manager,
    std::vector<std::pair<FeaturePerId *, double>> &cluster_candidates,
    std::deque<std::vector<Cluster>> &cluster, const int framecount) {

  for (auto &it_per_id : f_manager.feature) {

    it_per_id.clusterid = 0;

    if (it_per_id.start_frame + it_per_id.feature_per_frame.size() <
        framecount) {
      continue;
    }

    // check if feature is inside a moved cluster from old frames
    if (!cluster.empty()) {
      const auto puv(it_per_id.feature_per_frame.back().uv);
      const cv::Point p(puv.x(), puv.y());
      const int numclusters(cluster.size());
      int innumclusters(0);

      for (auto &cluster_frame : cluster) {
        for (auto &cluster_i : cluster_frame) {
          if (!cluster_i.convexhull.empty()) {
            const double result =
                cv::pointPolygonTest(cluster_i.convexhull, p, false);
            if (result > -1) {
              ++innumclusters;
            }
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
      cluster_candidates.push_back(std::make_pair(&it_per_id, 0));
    }
  }
}

void ClusterAlgorithm::moveCluster(std::deque<std::vector<Cluster>> &cluster) {

  // loop over frames of clusters
  for (auto &clusterframe : cluster) {
    std::vector<bool> invalid_cluster(clusterframe.size(), false);
    unsigned int local_currentclusterid(0);
    // loop over cluster in frame
    for (auto &cluster_i : clusterframe) {

      cv::Point2f cluster_i_center(cluster_i.center.x(), cluster_i.center.y());
      bool noclusterfound(true);
      // check if clustercenter is inside a cluster from last frame
      for (const auto &lastframe_cluster_i : cluster.back()) {

        const double result = cv::pointPolygonTest(
            lastframe_cluster_i.convexhull, cluster_i_center, false);
        if (result > -1) {
          Vector2d averageopticalflow(lastframe_cluster_i.averageopticalflow);
          // center inside cluster from last frame
          for (auto &point_i : cluster_i.convexhull) {
            // move with optical flow from cluster in last frame
            point_i.x += averageopticalflow.x() * 2.0;
            point_i.y += averageopticalflow.y() * 2.0;
          }
          noclusterfound = false;
          break;
        }
      }
      // invalidate cluster
      invalid_cluster[local_currentclusterid] = noclusterfound;
      ++local_currentclusterid;
    }

    // remove invalid clusters
    clusterframe.erase(
        std::remove_if(clusterframe.begin(), clusterframe.end(),
                       [&invalid_cluster, &clusterframe](Cluster const &i) {
                         return invalid_cluster.at(&i - clusterframe.data());
                       }),
        clusterframe.end());
  }
}

void ClusterAlgorithm::addInliers(FeatureManager &f_manager,
                                  const std::vector<Cluster> &new_cluster) {
  for (auto &cluster_i : new_cluster) {
    std::vector<cv::Point> convexclusterhull(cluster_i.convexhull);
    const double averageweight(cluster_i.averageweight);
    for (auto &it_per_id : f_manager.feature) {
      if (it_per_id.clusterid == 0) {

        const auto puv(it_per_id.feature_per_frame.back().uv);
        const cv::Point p(puv.x(), puv.y());
        const double result = cv::pointPolygonTest(convexclusterhull, p, false);

        if (result > -1) {
          // inside polygon
          it_per_id.clusterid = 2;
          it_per_id.weight = averageweight;
        }
      }
    }
  }
}

std::vector<Cluster> SimpleCluster::computecluster(
    FeatureManager &f_manager,
    std::vector<std::pair<FeaturePerId *, double>> &cluster_candidates,
    const int framecount) {

  // compute center and averageweight
  Vector2d center(0, 0);
  double averageweight(0.0);
  int num_in_cluster(cluster_candidates.size());
  for (auto candidate_i : cluster_candidates) {
    center += candidate_i.first->feature_per_frame.back().uv;
    averageweight += candidate_i.first->weight;
  }

  // cluster center for cluster-points with low weights
  Vector2d averageopticalflow(0.0, 0.0);
  std::vector<cv::Point> convexclusterhull;

  if (num_in_cluster) {
    center /= static_cast<double>(cluster_candidates.size());
    averageweight /= static_cast<double>(cluster_candidates.size());
    double averagedist(0.0);
    // compute distanecs to center
    for (auto &it_per_id : cluster_candidates) {
      const Vector2d dist(it_per_id.first->feature_per_frame.back().uv -
                          center);
      const double distscalar(dist.norm());
      averagedist += distscalar;
      it_per_id.second = distscalar;
    }

    averagedist /= static_cast<double>(cluster_candidates.size());

    // remove outliers
    cluster_candidates.erase(
        std::remove_if(cluster_candidates.begin(), cluster_candidates.end(),
                       [=](std::pair<FeaturePerId *, double> i) {
                         // remove from cluster if distance is bigger than 2
                         // times the averagedistance
                         bool outlier(i.second > 2.0 * averagedist);
                         if (outlier) {
                           i.first->clusterid = -1;
                         }
                         return outlier;
                       }),
        cluster_candidates.end());

    // compute new clustercenter and averageweight*
    center = Vector2d(0, 0);

    std::vector<cv::Point> inputarray;
    for (auto &it_per_id : cluster_candidates) {
      const auto puv = it_per_id.first->feature_per_frame.back().uv;
      inputarray.push_back(cv::Point(puv.x(), puv.y()));
      center += puv;
      averageweight += it_per_id.first->weight;
      averageopticalflow += it_per_id.first->feature_per_frame.back().velocity;
    }
    averageweight /= static_cast<double>(cluster_candidates.size());
    center /= static_cast<double>(cluster_candidates.size());

    // compute convexhull of this cluster
    cv::convexHull(inputarray, convexclusterhull);
  }
  Cluster temp_cluster;
  temp_cluster.center = center;
  temp_cluster.convexhull = convexclusterhull;
  temp_cluster.averageweight = averageweight;
  temp_cluster.averageopticalflow = averageopticalflow;
  std::vector<Cluster> tmp_clustervec;
  tmp_clustervec.push_back(temp_cluster);
  return tmp_clustervec;
}

std::vector<Cluster> DbscanCluster::computecluster(
    FeatureManager &f_manager,
    std::vector<std::pair<FeaturePerId *, double>> &cluster_candidates,
    const int framecount) {}
