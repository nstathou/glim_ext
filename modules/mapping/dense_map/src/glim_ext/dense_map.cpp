// Dense map export.
//
// Captures every odometry frame's full-resolution scan (requires keep_raw_points=true in config_ros.json),
// deskews it with the frame's IMU-rate trajectory into the IMU frame at frame->stamp, and on exit writes:
//   <dump>/<submap_id>/dense_scans.bin : per-frame local scans of that submap
//       repeated { int64 frame_id; float64 stamp; uint64 num_points; float32[num_points][4] (x, y, z, intensity) }
//       points are in the IMU frame at stamp; the frame's optimized pose is the matching line of traj_imu.txt
//   <dump>/dense_map.pcd : all scans transformed with the optimized poses (binary, x y z intensity)
// Optional voxel grids (0 = off, averaging xyz and intensity): scan_voxel_resolution per local scan (also affects
// dense_scans.bin), map_voxel_resolution over the merged world map.
#include <mutex>
#include <thread>
#include <fstream>
#include <unordered_map>

#include <spdlog/spdlog.h>
#include <gtsam_points/types/point_cloud_cpu.hpp>

#include <glim/util/config.hpp>
#include <glim/util/logging.hpp>
#include <glim/util/concurrent_vector.hpp>
#include <glim/util/extension_module.hpp>
#include <glim/common/cloud_deskewing.hpp>
#include <glim/odometry/callbacks.hpp>
#include <glim/mapping/callbacks.hpp>
#include <glim_ext/util/config_ext.hpp>

namespace glim {

class DenseMap : public ExtensionModule {
public:
  DenseMap() : logger(create_module_logger("dense_map")) {
    Config config(GlobalConfigExt::get_config_path("config_dense_map"));
    min_range = config.param<double>("dense_map", "min_range", 1.0);
    max_range = config.param<double>("dense_map", "max_range", 0.0);
    enable_cropbox_filter = config.param<bool>("dense_map", "enable_cropbox_filter", false);
    crop_bbox_min = config.param<Eigen::Vector3d>("dense_map", "crop_bbox_min", Eigen::Vector3d::Zero());
    crop_bbox_max = config.param<Eigen::Vector3d>("dense_map", "crop_bbox_max", Eigen::Vector3d::Zero());
    enable_outlier_removal = config.param<bool>("dense_map", "enable_outlier_removal", false);
    outlier_removal_k = config.param<int>("dense_map", "outlier_removal_k", 10);
    outlier_std_mul_factor = config.param<double>("dense_map", "outlier_std_mul_factor", 2.0);
    scan_voxel_resolution = config.param<double>("dense_map", "scan_voxel_resolution", 0.0);
    map_voxel_resolution = config.param<double>("dense_map", "map_voxel_resolution", 0.0);
    num_threads = config.param<int>("dense_map", "num_threads", 2);
    max_queue_size = config.param<int>("dense_map", "max_queue_size", 20);

    OdometryEstimationCallbacks::on_update_new_frame.add([this](const EstimationFrame::ConstPtr& frame) { input_queue.push_back(frame); });
    GlobalMappingCallbacks::on_update_submaps.add([this](const std::vector<SubMap::Ptr>& new_submaps) {
      std::lock_guard<std::mutex> lock(mutex);
      submaps = new_submaps;
    });

    thread = std::thread([this] { task(); });
    logger->info("ready (outlier_removal={} scan_voxel={} map_voxel={})", enable_outlier_removal, scan_voxel_resolution, map_voxel_resolution);
  }

  ~DenseMap() override { stop(); }

  // Throttles offline playback so queued raw scans cannot pile up in memory
  bool needs_wait() const override { return input_queue.size() > max_queue_size; }

  void at_exit(const std::string& dump_path) override {
    stop();
    std::lock_guard<std::mutex> lock(mutex);

    // Optimized frame poses, same formula as GlobalMapping::save() uses for traj_imu.txt
    std::vector<std::pair<const std::vector<Eigen::Vector4f>*, Eigen::Isometry3f>> placed;
    size_t total_points = 0;
    for (int i = 0; i < submaps.size(); i++) {
      const auto& submap = submaps[i];
      const Eigen::Isometry3d T_world_endpoint_L = submap->T_world_origin * submap->T_origin_endpoint_L;
      const Eigen::Isometry3d T_odom_imu0 = submap->frames.front()->T_world_imu;

      std::ofstream ofs(fmt::format("{}/{:06d}/dense_scans.bin", dump_path, i), std::ios::binary);
      for (const auto& frame : submap->frames) {
        const auto found = scans.find(frame->id);
        if (found == scans.end()) {
          continue;
        }

        const auto& points = found->second;
        const int64_t id = frame->id;
        const uint64_t num_points = points.size();
        ofs.write(reinterpret_cast<const char*>(&id), sizeof(id));
        ofs.write(reinterpret_cast<const char*>(&frame->stamp), sizeof(double));
        ofs.write(reinterpret_cast<const char*>(&num_points), sizeof(num_points));
        ofs.write(reinterpret_cast<const char*>(points.data()), sizeof(Eigen::Vector4f) * num_points);

        const Eigen::Isometry3d T_world_imu = T_world_endpoint_L * T_odom_imu0.inverse() * frame->T_world_imu;
        placed.emplace_back(&points, T_world_imu.cast<float>());
        total_points += num_points;
      }
    }
    logger->info("placed {} / {} captured scans ({} points)", placed.size(), scans.size(), total_points);

    const std::string pcd_path = dump_path + "/dense_map.pcd";
    std::ofstream ofs(pcd_path, std::ios::binary);
    const auto write_header = [&](size_t num_points) {
      ofs << "# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n";
      ofs << "WIDTH " << num_points << "\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS " << num_points << "\nDATA binary\n";
    };

    if (map_voxel_resolution > 0.0) {
      // Streaming voxel grid: average of xyz and intensity per voxel, without materializing the full-res world cloud
      struct Accum {
        float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        uint32_t count = 0;
      };
      std::unordered_map<uint64_t, Accum> voxels;
      const float inv_res = 1.0 / map_voxel_resolution;
      for (const auto& [points, T_world_imu] : placed) {
        for (const auto& pt : *points) {
          const Eigen::Vector3f p = T_world_imu * pt.head<3>();
          const Eigen::Array3i c = (p.array() * inv_res).floor().cast<int>();
          // 21 bits per axis: +-2^20 voxels, i.e. +-10 km at 1 cm
          const uint64_t key = (static_cast<uint64_t>(c.x() + (1 << 20)) & 0x1FFFFF) | ((static_cast<uint64_t>(c.y() + (1 << 20)) & 0x1FFFFF) << 21) |
                               ((static_cast<uint64_t>(c.z() + (1 << 20)) & 0x1FFFFF) << 42);
          auto& voxel = voxels[key];
          voxel.sum[0] += p.x();
          voxel.sum[1] += p.y();
          voxel.sum[2] += p.z();
          voxel.sum[3] += pt[3];
          voxel.count++;
        }
      }

      write_header(voxels.size());
      for (const auto& [key, voxel] : voxels) {
        const float mean[4] = {voxel.sum[0] / voxel.count, voxel.sum[1] / voxel.count, voxel.sum[2] / voxel.count, voxel.sum[3] / voxel.count};
        ofs.write(reinterpret_cast<const char*>(mean), sizeof(mean));
      }
      logger->info("map voxel grid {} m: {} -> {} points", map_voxel_resolution, total_points, voxels.size());
    } else {
      write_header(total_points);
      std::vector<Eigen::Vector4f> transformed;
      for (const auto& [points, T_world_imu] : placed) {
        transformed.resize(points->size());
        for (size_t j = 0; j < points->size(); j++) {
          transformed[j] << T_world_imu * (*points)[j].head<3>(), (*points)[j][3];
        }
        ofs.write(reinterpret_cast<const char*>(transformed.data()), sizeof(Eigen::Vector4f) * transformed.size());
      }
    }

    if (!ofs) {
      logger->error("failed to write {} (disk full?)", pcd_path);
    } else {
      logger->info("saved dense map to {}", pcd_path);
    }
  }

private:
  void stop() {
    input_queue.submit_end_of_data();
    if (thread.joinable()) {
      thread.join();
    }
  }

  void task() {
    while (const auto frame = input_queue.pop_wait()) {
      if (!(*frame)->raw_frame || !(*frame)->raw_frame->raw_points) {
        logger->warn("frame {} has no raw points. Set keep_raw_points=true in config_ros.json", (*frame)->id);
        continue;
      }
      if ((*frame)->imu_rate_trajectory.cols() < 2) {
        logger->warn("frame {} has no IMU-rate trajectory, skipping", (*frame)->id);
        continue;
      }

      auto points = process(*frame);
      std::lock_guard<std::mutex> lock(mutex);
      scans[(*frame)->id] = std::move(points);
    }
  }

  // Filter, deskew, and express the full-resolution scan in the IMU frame at raw_points->stamp
  std::vector<Eigen::Vector4f> process(const EstimationFrame::ConstPtr& frame) {
    const auto& raw = frame->raw_frame->raw_points;

    std::vector<int> indices;
    indices.reserve(raw->size());
    for (int i = 0; i < raw->size(); i++) {
      const Eigen::Vector3d p = raw->points[i].head<3>();
      const double range = p.norm();
      const bool in_crop_box = enable_cropbox_filter && (p.array() > crop_bbox_min.array()).all() && (p.array() < crop_bbox_max.array()).all();
      if (std::isfinite(range) && range >= min_range && (max_range <= 0.0 || range <= max_range) && !in_crop_box) {
        indices.push_back(i);
      }
    }

    auto cloud = std::make_shared<gtsam_points::PointCloudCPU>();
    std::vector<Eigen::Vector4d> points(indices.size());
    std::vector<double> times(indices.size());
    std::vector<double> intensities(indices.size(), 0.0);
    for (int i = 0; i < indices.size(); i++) {
      points[i] = raw->points[indices[i]];
      times[i] = raw->times[indices[i]];
      if (!raw->intensities.empty()) {
        intensities[i] = raw->intensities[indices[i]];
      }
    }
    cloud->add_points(points);
    cloud->add_times(times);
    cloud->add_intensities(intensities);
    cloud = gtsam_points::sort_by_time(cloud);  // CloudDeskewing requires time-ordered points

    // IMU-rate poses from scan start onward, time-ordered (FAST-LIO2 also stores a few samples just before the scan start)
    const auto& traj = frame->imu_rate_trajectory;
    std::vector<std::pair<double, Eigen::Isometry3d>> samples;
    for (int i = 0; i < traj.cols(); i++) {
      if (traj(0, i) < raw->stamp - 1e-6) {
        continue;
      }
      Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
      pose.translation() = traj.block<3, 1>(1, i);
      pose.linear() = Eigen::Quaterniond(traj(7, i), traj(4, i), traj(5, i), traj(6, i)).toRotationMatrix();
      samples.emplace_back(traj(0, i), pose);
    }
    std::sort(samples.begin(), samples.end(), [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

    std::vector<double> imu_times(samples.size());
    std::vector<Eigen::Isometry3d> imu_poses(samples.size());
    for (int i = 0; i < samples.size(); i++) {
      std::tie(imu_times[i], imu_poses[i]) = samples[i];
    }

    const Eigen::Isometry3d T_imu_lidar = frame->T_lidar_imu.inverse();
    auto deskewed = std::make_shared<gtsam_points::PointCloudCPU>(
      deskewing.deskew(T_imu_lidar, imu_times, imu_poses, raw->stamp, cloud->times_storage, cloud->points_storage));
    deskewed->add_intensities(cloud->intensities_storage);

    if (enable_outlier_removal) {
      deskewed = gtsam_points::remove_outliers(deskewed, outlier_removal_k, outlier_std_mul_factor, num_threads);
    }
    if (scan_voxel_resolution > 0.0) {
      deskewed = gtsam_points::voxelgrid_sampling(deskewed, scan_voxel_resolution, num_threads);  // averages xyz and intensity per voxel
    }

    std::vector<Eigen::Vector4f> result(deskewed->size());
    for (int i = 0; i < deskewed->size(); i++) {
      result[i] << (T_imu_lidar * deskewed->points[i]).head<3>().cast<float>(), static_cast<float>(deskewed->intensities[i]);
    }
    return result;
  }

private:
  double min_range;
  double max_range;
  bool enable_cropbox_filter;
  Eigen::Vector3d crop_bbox_min;
  Eigen::Vector3d crop_bbox_max;
  bool enable_outlier_removal;
  int outlier_removal_k;
  double outlier_std_mul_factor;
  double scan_voxel_resolution;
  double map_voxel_resolution;
  int num_threads;
  int max_queue_size;

  CloudDeskewing deskewing;
  ConcurrentVector<EstimationFrame::ConstPtr> input_queue;
  std::thread thread;

  std::mutex mutex;
  std::unordered_map<long, std::vector<Eigen::Vector4f>> scans;  // frame id -> local full-res scan (x, y, z, intensity)
  std::vector<SubMap::Ptr> submaps;

  std::shared_ptr<spdlog::logger> logger;
};

}  // namespace glim

extern "C" glim::ExtensionModule* create_extension_module() {
  return new glim::DenseMap();
}
