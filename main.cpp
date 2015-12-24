#include "pc2tsdf/pc2tsdf.h"
#include "detect_keypoints.h"
#include "ddd.h"
#include "ransacK.cpp"

#include <numeric>      // std::iota
#include <algorithm>    // std::sort
#include <iostream>     // std::cout
#include <functional>   // std::bind


bool sort_arr_compare(int a, int b, float* data) {
    return data[a]>data[b];
}

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

  // Find keypoints in first TUDF
  std::vector<std::vector<int>> keypoints1 = detect_keypoints(scene_tsdf1, x_dim1, y_dim1, z_dim1, 0.2f, 1.0f, 5, 100.0f);

  // Filter out keypoints too close to the bounding box of the voxel volume
  int local_patch_radius = 15;
  std::vector<std::vector<int>> valid_keypoints1;
  for (int i = 0; i < keypoints1.size(); i++)
    if (keypoints1[i][0] - local_patch_radius >= 0 && keypoints1[i][0] + local_patch_radius < x_dim1 &&
        keypoints1[i][1] - local_patch_radius >= 0 && keypoints1[i][1] + local_patch_radius < y_dim1 &&
        keypoints1[i][2] - local_patch_radius >= 0 && keypoints1[i][2] + local_patch_radius < z_dim1)
      valid_keypoints1.push_back(keypoints1[i]);

  // Compute ddd features from keypoints
  std::vector<std::vector<float>> feat1;  
  feat1 = ddd_get_keypoint_feat(scene_tsdf1, x_dim1, y_dim1, z_dim1, valid_keypoints1, local_patch_radius);

  // Convert valid keypoints from grid to world coordinates
  std::vector<std::vector<float>> world_keypoints1;
  for (int i = 0; i < valid_keypoints1.size(); i++) {
    std::vector<float> tmp_keypoint;
    tmp_keypoint.push_back((float) valid_keypoints1[i][0] * voxelSize + tsdf1.worldOrigin[0]);
    tmp_keypoint.push_back((float) valid_keypoints1[i][1] * voxelSize + tsdf1.worldOrigin[1]);
    tmp_keypoint.push_back((float) valid_keypoints1[i][2] * voxelSize + tsdf1.worldOrigin[2]);
    world_keypoints1.push_back(tmp_keypoint);
  }

  ///////////////////////////////////////////////////////////////////

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

  // Find keypoints in second TUDF
  std::vector<std::vector<int>> keypoints2 = detect_keypoints(scene_tsdf2, x_dim2, y_dim2, z_dim2, 0.2f, 1.0f, 5, 100.0f);

  // Filter out keypoints too close to the bounding box of the voxel volume
  local_patch_radius = 15;
  std::vector<std::vector<int>> valid_keypoints2;
  for (int i = 0; i < keypoints2.size(); i++)
    if (keypoints2[i][0] - local_patch_radius >= 0 && keypoints2[i][0] + local_patch_radius < x_dim2 &&
        keypoints2[i][1] - local_patch_radius >= 0 && keypoints2[i][1] + local_patch_radius < y_dim2 &&
        keypoints2[i][2] - local_patch_radius >= 0 && keypoints2[i][2] + local_patch_radius < z_dim2)
      valid_keypoints2.push_back(keypoints2[i]);

  // Compute ddd features from keypoints
  std::vector<std::vector<float>> feat2;  
  feat2 = ddd_get_keypoint_feat(scene_tsdf2, x_dim2, y_dim2, z_dim2, valid_keypoints2, local_patch_radius);

  // Convert valid keypoints from grid to world coordinates
  std::vector<std::vector<float>> world_keypoints2;
  for (int i = 0; i < valid_keypoints2.size(); i++) {
    std::vector<float> tmp_keypoint;
    tmp_keypoint.push_back((float) valid_keypoints2[i][0] * voxelSize + tsdf2.worldOrigin[0]);
    tmp_keypoint.push_back((float) valid_keypoints2[i][1] * voxelSize + tsdf2.worldOrigin[1]);
    tmp_keypoint.push_back((float) valid_keypoints2[i][2] * voxelSize + tsdf2.worldOrigin[2]);
    world_keypoints2.push_back(tmp_keypoint);
  }

  ///////////////////////////////////////////////////////////////////
 
  std::vector<std::vector<float>> score_matrix;  
  score_matrix = ddd_compare_feat(feat1, feat2);

  // for (int i = 0; i < feat2.size(); i++) {
  //   std::cout << i << " " << score_matrix[0][i] << std::endl;
  // }
  float score_threshold = 0.5;

  std::vector<std::vector<int>> match_rank;
  for (int i = 0; i < feat1.size(); i++) {
    std::vector<float> tmp_score_vect = score_matrix[i];
    float* tmp_score_vect_arr = &tmp_score_vect[0];
    int* tmp_score_idx = new int[tmp_score_vect.size()];
    std::iota(tmp_score_idx, tmp_score_idx + tmp_score_vect.size(), 0);
    std::sort(tmp_score_idx, tmp_score_idx + tmp_score_vect.size(), std::bind(sort_arr_compare, std::placeholders::_1, std::placeholders::_2, tmp_score_vect_arr));
    

    std::vector<int> tmp_score_rank;
    for (int j = 0; j < feat2.size(); j++)
      if (tmp_score_vect_arr[tmp_score_idx[j]] > score_threshold)
        tmp_score_rank.push_back(tmp_score_idx[j]);

    std::cout << tmp_score_rank.size() << std::endl;
    match_rank.push_back(tmp_score_rank);

    // std::cout << "data:   "; for (int i=0; i<10; i++) std::cout << tmp_score_vect_arr[i] << " "; std::cout << "\n";
    // std::cout << "index:  "; for (int i=0; i<10; i++) std::cout << tmp_score_rank[i] << " "; std::cout << "\n";
    // std::cout << "sorted: "; for (int i=0; i<10; i++) std::cout << tmp_score_vect_arr[tmp_score_idx[i]] << " "; std::cout << "\n";
    // std::cout << std::endl;

  }

  for (int i = 0; i < feat1.size(); i++) {
    std::cout << i << " ";
    for (int j = 0; j < match_rank[i].size(); j++)
      std::cout << match_rank[i][j] << " ";
    std::cout << std::endl;
  }

  float* Rt = new float[12];
  ransacfitRt(world_keypoints2, world_keypoints1, match_rank, 10, 100000, 0.05f, Rt);























  // Convert valid keypoints from grid to world coordinates


  // // for (int i = 0; i < keypoints.size(); i++) {
  // //   std::cout << keypoints[i][0] << " " << keypoints[i][1] << " " << keypoints[i][2] << std::endl;
  // // }

  // FILE *fp = fopen("test1.ply", "w");
  // fprintf(fp, "ply\n");
  // fprintf(fp, "format binary_little_endian 1.0\n");
  // fprintf(fp, "element vertex %d\n", (int)world_keypoints1.size());
  // fprintf(fp, "property float x\n");
  // fprintf(fp, "property float y\n");
  // fprintf(fp, "property float z\n");
  // fprintf(fp, "property uchar red\n");
  // fprintf(fp, "property uchar green\n");
  // fprintf(fp, "property uchar blue\n");
  // fprintf(fp, "end_header\n");
  // for (int i = 0; i < world_keypoints1.size(); i++) {
  //   // std::cout << keypoints[i][0] << " " << keypoints[i][1] << " " << keypoints[i][2] << std::endl;
  //   float float_x = (float) world_keypoints1[i][0];
  //   float float_y = (float) world_keypoints1[i][1];
  //   float float_z = (float) world_keypoints1[i][2];
  //   fwrite(&float_x, sizeof(float), 1, fp);
  //   fwrite(&float_y, sizeof(float), 1, fp);
  //   fwrite(&float_z, sizeof(float), 1, fp);
  //   uchar r = (uchar)250;
  //   uchar g = (uchar)0;
  //   uchar b = (uchar)0;
  //   fwrite(&r, sizeof(uchar), 1, fp);
  //   fwrite(&g, sizeof(uchar), 1, fp);
  //   fwrite(&b, sizeof(uchar), 1, fp);
  // }
  // fclose(fp);

  // fp = fopen("test2.ply", "w");
  // fprintf(fp, "ply\n");
  // fprintf(fp, "format binary_little_endian 1.0\n");
  // fprintf(fp, "element vertex %d\n", (int)world_keypoints2.size());
  // fprintf(fp, "property float x\n");
  // fprintf(fp, "property float y\n");
  // fprintf(fp, "property float z\n");
  // fprintf(fp, "property uchar red\n");
  // fprintf(fp, "property uchar green\n");
  // fprintf(fp, "property uchar blue\n");
  // fprintf(fp, "end_header\n");
  // for (int i = 0; i < world_keypoints2.size(); i++) {
  //   // std::cout << keypoints[i][0] << " " << keypoints[i][1] << " " << keypoints[i][2] << std::endl;
  //   float float_x = (float) world_keypoints2[i][0];
  //   float float_y = (float) world_keypoints2[i][1];
  //   float float_z = (float) world_keypoints2[i][2];
  //   fwrite(&float_x, sizeof(float), 1, fp);
  //   fwrite(&float_y, sizeof(float), 1, fp);
  //   fwrite(&float_z, sizeof(float), 1, fp);
  //   uchar r = (uchar)250;
  //   uchar g = (uchar)0;
  //   uchar b = (uchar)0;
  //   fwrite(&r, sizeof(uchar), 1, fp);
  //   fwrite(&g, sizeof(uchar), 1, fp);
  //   fwrite(&b, sizeof(uchar), 1, fp);
  // }
  // fclose(fp);


  // std::cout << keypoints.size() << std::endl;


  // >/dev/null







  return 0;
}
