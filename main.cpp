#include "pc2tsdf/pc2tsdf.h"
#include "detect_keypoints.h"
#include "ddd.h"

int main() {

  const float voxelSize = 0.01f;
  const float truncationRadius = 0.05f;

  ///////////////////////////////////////////////////////////////////

  // Load first point cloud and save to TUDF grid data
  auto cloud1 = PointCloudIOf::loadFromFile("cloud_bin_0.ply");
  pc2tsdf::TSDF tsdf1;
  pc2tsdf::makeTSDF(cloud1, voxelSize, truncationRadius, tsdf1);
  // tsdf1.saveBinary("testFragmentFixed.tsdf");

  // Convert TUDF grid data to float array
  int x_dim1 = (int)tsdf1.data.getDimensions().x;
  int y_dim1 = (int)tsdf1.data.getDimensions().y;
  int z_dim1 = (int)tsdf1.data.getDimensions().z;
  float *scene_tsdf1 = new float[x_dim1 * y_dim1 * z_dim1];
  for (int i = 0; i < x_dim1 * y_dim1 * z_dim1; i++)
    scene_tsdf1[i] = 1.0;
  for (int z = 0; z < (int)tsdf1.data.getDimensions().z; z++)
    for (int y = 0; y < (int)tsdf1.data.getDimensions().y; y++)
      for (int x = 0; x < (int)tsdf1.data.getDimensions().x; x++)
        scene_tsdf1[z * y_dim1 * x_dim1 + y * x_dim1 + x] = tsdf1.data(x, y, z) / truncationRadius;

  // Load second point cloud and save to TUDF
  auto cloud2 = PointCloudIOf::loadFromFile("cloud_bin_1.ply");
  pc2tsdf::TSDF tsdf2;
  pc2tsdf::makeTSDF(cloud2, voxelSize, truncationRadius, tsdf2);

  // Convert TUDF grid data to float array
  int x_dim2 = (int)tsdf2.data.getDimensions().x;
  int y_dim2 = (int)tsdf2.data.getDimensions().y;
  int z_dim2 = (int)tsdf2.data.getDimensions().z;
  float *scene_tsdf2 = new float[x_dim2 * y_dim2 * z_dim2];
  for (int i = 0; i < x_dim2 * y_dim2 * z_dim2; i++)
    scene_tsdf2[i] = 1.0;
  for (int z = 0; z < (int)tsdf2.data.getDimensions().z; z++)
    for (int y = 0; y < (int)tsdf2.data.getDimensions().y; y++)
      for (int x = 0; x < (int)tsdf2.data.getDimensions().x; x++)
        scene_tsdf2[z * y_dim2 * x_dim2 + y * x_dim2 + x] = tsdf2.data(x, y, z) / truncationRadius;

  ///////////////////////////////////////////////////////////////////

  float k_match_score_thresh = 0.5f;
  float ransac_k = 10; // RANSAC over top-k > k_match_score_thresh
  float max_ransac_iter = 1000000;
  float ransac_inlier_thresh = 0.01f;
  float* Rt = new float[12]; // Contains rigid transform matrix
  align2tsdf(scene_tsdf1, x_dim1, y_dim1, z_dim1, tsdf1.worldOrigin[0], tsdf1.worldOrigin[1], tsdf1.worldOrigin[2],
             scene_tsdf2, x_dim2, y_dim2, z_dim2, tsdf2.worldOrigin[0], tsdf2.worldOrigin[1], tsdf2.worldOrigin[2], 
             voxelSize, k_match_score_thresh, ransac_k, max_ransac_iter, ransac_inlier_thresh, Rt);

  ///////////////////////////////////////////////////////////////////

  // Apply Rt to second point cloud and align it to first
  for (int i = 0; i < cloud2.m_points.size(); i++) {
    vec3f tmp_point;
    tmp_point.x = Rt[0] * cloud2.m_points[i].x + Rt[1] * cloud2.m_points[i].y + Rt[2] * cloud2.m_points[i].z;
    tmp_point.y = Rt[4] * cloud2.m_points[i].x + Rt[5] * cloud2.m_points[i].y + Rt[6] * cloud2.m_points[i].z;
    tmp_point.z = Rt[8] * cloud2.m_points[i].x + Rt[9] * cloud2.m_points[i].y + Rt[10] * cloud2.m_points[i].z;
    tmp_point.x = tmp_point.x + Rt[3];
    tmp_point.y = tmp_point.y + Rt[7];
    tmp_point.z = tmp_point.z + Rt[11];
    cloud2.m_points[i] = tmp_point;
  }

  // Print out both point clouds
  std::string pcfile1 = "test1.ply";
  PointCloudIOf::saveToFile(pcfile1, cloud1.m_points);
  std::string pcfile2 = "test2.ply";
  PointCloudIOf::saveToFile(pcfile2, cloud2.m_points);
  // >/dev/null

  return 0;
}
