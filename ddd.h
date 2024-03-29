#include <vector>
#include <numeric>      
#include <algorithm>   
#include <iostream>     
#include <functional> 
#include "ransacK.cpp"  
// #include "util.h"

const bool ddd_verbose = true;

///////////////////////////////////////////////////////////////////////
// Given a location (x, y, z) in the voxel volume, return a local voxel
// patch volume around (x, y, z). Given radius r, the patch dimensions
// are (2r+1)x(2r+1)x(2r+1)
//
// Parameters:
//   volume - dense array of floats to represent voxel volume
//            volume[z*y_dim*x_dim + y*x_dim + x] --> volume(x,y,z)
//   x_dim, y_dim, z_dim - voxel volume size in X,Y,Z dimensions
//   x, y, z - voxel location as center of local patch volume
//   radius  - radius of local voxel patch volume
//   local_patch - dense array of floats to represent local patch volume
//
// Local patch volume is saved into parameter "local_patch"
// Returns true if local patch volume was successfully retrieved
//
// Copyright (c) 2015 Andy Zeng, Princeton University
void get_keypoint_volume(float* volume, int x_dim, int y_dim, int z_dim, int x, int y, int z, int radius, float* local_patch) {

  // Assign values to local patch volume
  int patch_dim = radius * 2 + 1;
  for (int k = 0; k < patch_dim; k++)
    for (int j = 0; j < patch_dim; j++)
      for (int i = 0; i < patch_dim; i++) {
        int vx = x - radius + i;
        int vy = y - radius + j;
        int vz = z - radius + k;
        float value = volume[vz * y_dim * x_dim + vy * x_dim + vx];
        local_patch[k * patch_dim * patch_dim + j * patch_dim + i] = value;
      }
}

void json_data_location_replace(std::string src, std::string dst, std::string srch_str, std::string rplc_str) {
  std::ifstream in(src);
  std::ofstream out(dst);
  std::string line;
  size_t len = srch_str.length();

  if (!in) {
    std::cerr << "Could not open " << src << "\n";
    return;
  }

  if (!out) {
    std::cerr << "Could not open " << dst << "\n";
    return;
  }

  while (getline(in, line)) {
    while (true) {
      size_t pos = line.find(srch_str);
      if (pos != std::string::npos)
        line.replace(pos, len, rplc_str);
      break;
    }
    out << line << '\n';
  }
}

///////////////////////////////////////////////////////////////////////
std::vector<std::vector<float>> ddd_get_keypoint_feat(float* volume, int x_dim, int y_dim, int z_dim, std::vector<std::vector<int>> &keypoints, int patch_radius, bool is_verbose) {

  // Create random hash ID for ddd instance
  std::string instance_id = gen_rand_str(16);

  // Init tensor files for marvin
  std::string data_tensor_filename = "TMPdata_" + instance_id + ".tensor";
  std::string label_tensor_filename = "TMPlabels_" + instance_id + ".tensor";
  FILE *data_tensor_fp = fopen(data_tensor_filename.c_str(), "w");
  FILE *label_tensor_fp = fopen(label_tensor_filename.c_str(), "w");

  // Write data header
  int num_keypoints = keypoints.size();
  int patch_dim = patch_radius * 2 + 1;
  uint8_t tmp_typeid = (uint8_t)1;
  fwrite((void*)&tmp_typeid, sizeof(uint8_t), 1, data_tensor_fp);
  uint32_t tmp_sizeof = (uint32_t)4;
  fwrite((void*)&tmp_sizeof, sizeof(uint32_t), 1, data_tensor_fp);
  uint32_t tmp_strlen = (uint32_t)4;
  fwrite((void*)&tmp_strlen, sizeof(uint32_t), 1, data_tensor_fp);
  fprintf(data_tensor_fp, "data");
  uint32_t tmp_data_dim = (uint32_t)5;
  fwrite((void*)&tmp_data_dim, sizeof(uint32_t), 1, data_tensor_fp);
  uint32_t tmp_size = (uint32_t)num_keypoints;
  fwrite((void*)&tmp_size, sizeof(uint32_t), 1, data_tensor_fp);
  uint32_t tmp_data_chan = (uint32_t)1;
  fwrite((void*)&tmp_data_chan, sizeof(uint32_t), 1, data_tensor_fp);
  uint32_t tmp_volume_dim = (uint32_t)patch_dim;
  fwrite((void*)&tmp_volume_dim, sizeof(uint32_t), 1, data_tensor_fp);
  fwrite((void*)&tmp_volume_dim, sizeof(uint32_t), 1, data_tensor_fp);
  fwrite((void*)&tmp_volume_dim, sizeof(uint32_t), 1, data_tensor_fp);

  // Write label header
  tmp_typeid = (uint8_t)1;
  fwrite((void*)&tmp_typeid, sizeof(uint8_t), 1, label_tensor_fp);
  tmp_sizeof = (uint32_t)4;
  fwrite((void*)&tmp_sizeof, sizeof(uint32_t), 1, label_tensor_fp);
  tmp_strlen = (uint32_t)6;
  fwrite((void*)&tmp_strlen, sizeof(uint32_t), 1, label_tensor_fp);
  fprintf(label_tensor_fp, "labels");
  tmp_data_dim = (uint32_t)5;
  fwrite((void*)&tmp_data_dim, sizeof(uint32_t), 1, label_tensor_fp);
  tmp_size = (uint32_t)num_keypoints;
  fwrite((void*)&tmp_size, sizeof(uint32_t), 1, label_tensor_fp);
  tmp_data_chan = (uint32_t)1;
  fwrite((void*)&tmp_data_chan, sizeof(uint32_t), 1, label_tensor_fp);
  tmp_volume_dim = (uint32_t)1;
  fwrite((void*)&tmp_volume_dim, sizeof(uint32_t), 1, label_tensor_fp);
  fwrite((void*)&tmp_volume_dim, sizeof(uint32_t), 1, label_tensor_fp);
  fwrite((void*)&tmp_volume_dim, sizeof(uint32_t), 1, label_tensor_fp);

  // Extract local volume comparisons and save to tensor files
  for (int i = 0; i < keypoints.size(); i++) {
    if (ddd_verbose)
      std::cout << "Loading keypoint: " << i + 1 << "/" << num_keypoints << ": " << keypoints[i][0] << " " << keypoints[i][1] << " " << keypoints[i][2] << std::endl;
    // Extract local patch volume
    float *local_patch = new float[patch_dim * patch_dim * patch_dim];
    get_keypoint_volume(volume, x_dim, y_dim, z_dim, keypoints[i][0], keypoints[i][1], keypoints[i][2], patch_radius, local_patch);
    // Take absolute value of tsdf
    for (int k = 0; k < patch_dim * patch_dim * patch_dim; k++) {
      local_patch[k] = std::abs(local_patch[k]);
    }
    // Write local tsdf volume to data tensor file
    fwrite(local_patch, sizeof(float), patch_dim * patch_dim * patch_dim, data_tensor_fp);
    // Write dummy label to label tensor file
    float tmp_label = 0;
    fwrite(&tmp_label, sizeof(float), 1, label_tensor_fp);

    delete [] local_patch;
  }

  fclose(label_tensor_fp);
  fclose(data_tensor_fp);

  // Marvin params
  std::string feat_tensor_filename = "TMPfeat_" + instance_id + ".tensor";
  std::string model_filename = "ddd/dddnet.marvin";
  std::string architecture_filename = "ddd/featnet.json";
  std::string new_architecture_filename = "ddd/featnet_" + instance_id + ".json";
  std::string marvin_filename = "ddd/marvin";
  std::string cuda_lib_dir = "/usr/local/cuda/lib64";

  // Create custom marvin json file to read from certain file locations
  json_data_location_replace(architecture_filename, new_architecture_filename, ".tensor", "_" + instance_id + ".tensor");

  // Run marvin to get tensor file of feature vectors
  // sys_command("rm " + feat_tensor_filename);
  if (is_verbose)
    sys_command("export LD_LIBRARY_PATH=LD_LIBRARY_PATH:" + cuda_lib_dir + "; ./" + marvin_filename + " test " + new_architecture_filename + " " + model_filename + " feat " + feat_tensor_filename);
  else
    sys_command("export LD_LIBRARY_PATH=LD_LIBRARY_PATH:" + cuda_lib_dir + "; ./" + marvin_filename + " test " + new_architecture_filename + " " + model_filename + " feat " + feat_tensor_filename + " >/dev/null");
  sys_command("rm " + data_tensor_filename);
  sys_command("rm " + label_tensor_filename);
  sys_command("rm " + new_architecture_filename);

  // Read feature vectors from tensor file
  int feat_dim = 2048;
  std::vector<std::vector<float>> feat;
  std::ifstream inFile(feat_tensor_filename, std::ios::binary | std::ios::in);
  int header_bytes = (1 + 4 + 4) + (4) + (4 + 4 + 4 + 4 + 4);
  inFile.seekg(size_t(header_bytes));
  float *feat_raw = new float[num_keypoints * feat_dim];
  inFile.read((char*)feat_raw, num_keypoints * feat_dim * sizeof(float));
  inFile.close();
  for (int i = 0; i < num_keypoints; i++) {
    std::vector<float> tmp_feat;
    for (int j = 0; j < feat_dim; j++)
      tmp_feat.push_back(feat_raw[i * feat_dim + j]);
    feat.push_back(tmp_feat);
  }
  sys_command("rm " + feat_tensor_filename);

  delete [] feat_raw;

  return feat;
}

///////////////////////////////////////////////////////////////////////
void ddd_compare_feat(std::vector<std::vector<float>> &feat1, std::vector<std::vector<float>> &feat2, std::vector<std::vector<float>>* score_matrix, bool is_verbose) {

  int feat_dim = 2048;
  int num_cases = feat1.size() * feat2.size();

  // Create random hash ID for ddd instance
  std::string instance_id = gen_rand_str(16);

  // Init tensor files for marvin
  std::string data_tensor_filename = "TMPdata_" + instance_id + ".tensor";
  std::string label_tensor_filename = "TMPlabels_" + instance_id + ".tensor";
  FILE *data_tensor_fp = fopen(data_tensor_filename.c_str(), "w");
  FILE *label_tensor_fp = fopen(label_tensor_filename.c_str(), "w");

  // Write data header
  uint8_t tmp_typeid = (uint8_t)1;
  fwrite((void*)&tmp_typeid, sizeof(uint8_t), 1, data_tensor_fp);
  uint32_t tmp_sizeof = (uint32_t)4;
  fwrite((void*)&tmp_sizeof, sizeof(uint32_t), 1, data_tensor_fp);
  uint32_t tmp_strlen = (uint32_t)4;
  fwrite((void*)&tmp_strlen, sizeof(uint32_t), 1, data_tensor_fp);
  fprintf(data_tensor_fp, "data");
  uint32_t tmp_data_dim = (uint32_t)5;
  fwrite((void*)&tmp_data_dim, sizeof(uint32_t), 1, data_tensor_fp);
  uint32_t tmp_size = (uint32_t)num_cases;
  fwrite((void*)&tmp_size, sizeof(uint32_t), 1, data_tensor_fp);
  uint32_t tmp_data_chan = (uint32_t)feat_dim * 2;
  fwrite((void*)&tmp_data_chan, sizeof(uint32_t), 1, data_tensor_fp);
  uint32_t tmp_volume_dim = (uint32_t)1;
  fwrite((void*)&tmp_volume_dim, sizeof(uint32_t), 1, data_tensor_fp);
  fwrite((void*)&tmp_volume_dim, sizeof(uint32_t), 1, data_tensor_fp);
  fwrite((void*)&tmp_volume_dim, sizeof(uint32_t), 1, data_tensor_fp);

  // Write label header
  tmp_typeid = (uint8_t)1;
  fwrite((void*)&tmp_typeid, sizeof(uint8_t), 1, label_tensor_fp);
  tmp_sizeof = (uint32_t)4;
  fwrite((void*)&tmp_sizeof, sizeof(uint32_t), 1, label_tensor_fp);
  tmp_strlen = (uint32_t)6;
  fwrite((void*)&tmp_strlen, sizeof(uint32_t), 1, label_tensor_fp);
  fprintf(label_tensor_fp, "labels");
  tmp_data_dim = (uint32_t)5;
  fwrite((void*)&tmp_data_dim, sizeof(uint32_t), 1, label_tensor_fp);
  tmp_size = (uint32_t)num_cases;
  fwrite((void*)&tmp_size, sizeof(uint32_t), 1, label_tensor_fp);
  tmp_data_chan = (uint32_t)1;
  fwrite((void*)&tmp_data_chan, sizeof(uint32_t), 1, label_tensor_fp);
  tmp_volume_dim = (uint32_t)1;
  fwrite((void*)&tmp_volume_dim, sizeof(uint32_t), 1, label_tensor_fp);
  fwrite((void*)&tmp_volume_dim, sizeof(uint32_t), 1, label_tensor_fp);
  fwrite((void*)&tmp_volume_dim, sizeof(uint32_t), 1, label_tensor_fp);

  // Save feature vectors to tensor file
  for (int i = 0; i < feat1.size(); i++) {
    for (int j = 0; j < feat2.size(); j++) {
      if (ddd_verbose)
        std::cout << "Loading keypoint comparison: " << i*feat2.size() + j << "/" << feat1.size() * feat2.size() - 1 << std::endl;
      // Save feature vector 2
      fwrite(&feat2[j][0], sizeof(float), feat2[j].size(), data_tensor_fp);
      // Save feature vector 1
      fwrite(&feat1[i][0], sizeof(float), feat1[i].size(), data_tensor_fp);
      float tmp_label = 0;
      fwrite(&tmp_label, sizeof(float), 1, label_tensor_fp);
    }
  }

  fclose(label_tensor_fp);
  fclose(data_tensor_fp);

  // Marvin params
  std::string prob_tensor_filename = "TMPprob_" + instance_id + ".tensor";
  std::string model_filename = "ddd/dddnet.marvin";
  std::string architecture_filename = "ddd/metricnet.json";
  std::string new_architecture_filename = "ddd/metricnet_" + instance_id + ".json";
  std::string marvin_filename = "ddd/marvin";
  std::string cuda_lib_dir = "/usr/local/cuda/lib64";

  // Create custom marvin json file to read from certain file locations
  json_data_location_replace(architecture_filename, new_architecture_filename, ".tensor", "_" + instance_id + ".tensor");

  // Run marvin to get tensor file of match scores
  model_filename = "ddd/dddnet.marvin";
  // sys_command("rm " + prob_tensor_filename);
  if (is_verbose)
    sys_command("export LD_LIBRARY_PATH=LD_LIBRARY_PATH:" + cuda_lib_dir + "; ./" + marvin_filename + " test " + new_architecture_filename + " " + model_filename + " prob " + prob_tensor_filename);
  else
    sys_command("export LD_LIBRARY_PATH=LD_LIBRARY_PATH:" + cuda_lib_dir + "; ./" + marvin_filename + " test " + new_architecture_filename + " " + model_filename + " prob " + prob_tensor_filename + " >/dev/null");
  sys_command("rm " + data_tensor_filename);
  sys_command("rm " + label_tensor_filename);
  sys_command("rm " + new_architecture_filename);

  // Save match scores to matrix 
  std::ifstream inFile(prob_tensor_filename.c_str(), std::ios::binary | std::ios::in);
  int header_bytes = (1 + 4 + 4) + (4) + (4 + 4 + 4 + 4 + 4);
  inFile.seekg(size_t(header_bytes));
  float *labels_raw = new float[num_cases * 2];
  inFile.read((char*)labels_raw, num_cases * 2 * sizeof(float));
  inFile.close();
  float *match_scores = new float[num_cases];
  for (int i = 0; i < num_cases; i++)
    match_scores[i] = labels_raw[i * 2 + 1];
  for (int i = 0; i < feat1.size(); i++) {
    std::vector<float> tmp_score_vect;
    for (int j = 0; j < feat2.size(); j++)
      tmp_score_vect.push_back(match_scores[i * feat2.size() + j]);
    score_matrix->push_back(tmp_score_vect);
  }

  delete [] labels_raw;
  delete [] match_scores;

  sys_command("rm " + prob_tensor_filename);
}

///////////////////////////////////////////////////////////////////////
bool sort_arr_desc_compare(int a, int b, float* data) {
    return data[a]>data[b];
}

///////////////////////////////////////////////////////////////////////
std::vector<std::vector<int>> detect_keypoints_filtered(float* scene_tsdf, int x_dim, int y_dim, int z_dim) {
    // Find keypoints in TUDF
    std::vector<std::vector<int>> keypoints = detect_keypoints(scene_tsdf, x_dim, y_dim, z_dim, 0.2f, 1.0f, 5, 100.0f);

    // Filter out keypoints too close to the bounding box of the voxel volume
    int local_patch_radius = 15;
    std::vector<std::vector<int>> valid_keypoints;
    for (int i = 0; i < keypoints.size(); i++)
        if (keypoints[i][0] - local_patch_radius >= 0 && keypoints[i][0] + local_patch_radius < x_dim &&
            keypoints[i][1] - local_patch_radius >= 0 && keypoints[i][1] + local_patch_radius < y_dim &&
            keypoints[i][2] - local_patch_radius >= 0 && keypoints[i][2] + local_patch_radius < z_dim)
            valid_keypoints.push_back(keypoints[i]);
    return valid_keypoints;
}

///////////////////////////////////////////////////////////////////////
void ddd_compare_feature_cloud(std::vector<std::vector<float>> &feat1, std::vector<std::vector<float>>* score_matrix1,
                               std::vector<std::vector<float>> &feat2, std::vector<std::vector<float>>* score_matrix2) {
    // Compare feature vectors and compute score matrix
    tic();
    ddd_compare_feat(feat1, feat2, score_matrix1, ddd_verbose);
    std::cout << "Comparing keypoint features from both TSDF volumes. ";
    toc();    

    // Inversely compare feature vectors and compute score matrix
    tic();
    ddd_compare_feat(feat2, feat1, score_matrix2, ddd_verbose);
    std::cout << "Comparing *flipped* keypoint features from both TSDF volumes. ";
    toc();
}

///////////////////////////////////////////////////////////////////////
void ddd_align_feature_cloud(const std::vector<std::vector<float>> &world_keypoints1, std::vector<std::vector<float>> &feat1, const std::vector<std::vector<float>> &score_matrix1,
                             const std::vector<std::vector<float>> &world_keypoints2, std::vector<std::vector<float>> &feat2, const std::vector<std::vector<float>> &score_matrix2,
                             float voxelSize, float k_match_score_thresh, float ransac_k, float max_ransac_iter, float ransac_thresh, float* Rt) {

    // For each keypoint from first set, find indices of all keypoints
    // in second set with score > k_match_score_thresh
    std::vector<std::vector<int>> match_rank1;
    for (int i = 0; i < feat1.size(); i++) {
        // Sort score vector in descending fashion
        std::vector<float> tmp_score_vect = score_matrix1[i];
        float* tmp_score_vect_arr = tmp_score_vect.data();
        int* tmp_score_idx = new int[tmp_score_vect.size()];
        std::iota(tmp_score_idx, tmp_score_idx + tmp_score_vect.size(), 0);
        std::sort(tmp_score_idx, tmp_score_idx + tmp_score_vect.size(), std::bind(sort_arr_desc_compare, std::placeholders::_1, std::placeholders::_2, tmp_score_vect_arr));
        std::vector<int> tmp_score_rank;
        for (int j = 0; j < feat2.size(); j++)
            if (tmp_score_vect_arr[tmp_score_idx[j]] > k_match_score_thresh)
                tmp_score_rank.push_back(tmp_score_idx[j]);
        // std::cout << tmp_score_rank.size() << std::endl;
        match_rank1.push_back(tmp_score_rank);
        delete [] tmp_score_idx;
    }

    // For each keypoint from second set, find indices of all keypoints
    // in first set with score > k_match_score_thresh
    std::vector<std::vector<int>> match_rank2;
    for (int i = 0; i < feat2.size(); i++) {
        // Sort score vector in descending fashion
        std::vector<float> tmp_score_vect = score_matrix2[i];
        float* tmp_score_vect_arr = tmp_score_vect.data();
        int* tmp_score_idx = new int[tmp_score_vect.size()];
        std::iota(tmp_score_idx, tmp_score_idx + tmp_score_vect.size(), 0);
        std::sort(tmp_score_idx, tmp_score_idx + tmp_score_vect.size(), std::bind(sort_arr_desc_compare, std::placeholders::_1, std::placeholders::_2, tmp_score_vect_arr));
        std::vector<int> tmp_score_rank;
        for (int j = 0; j < feat1.size(); j++)
            if (tmp_score_vect_arr[tmp_score_idx[j]] > k_match_score_thresh)
                tmp_score_rank.push_back(tmp_score_idx[j]);
        // std::cout << tmp_score_rank.size() << std::endl;
        match_rank2.push_back(tmp_score_rank);
        delete [] tmp_score_idx;
    }

    // Finalize match matrix (indices) unofficial reflexive property
    // A pair of points (with feature vectors f1 and f2) match iff
    // ddd(f1,f2) > threshold && ddd(f2,f1) > threshold
    std::vector<std::vector<int>> match_idx;
    for (int i = 0; i < feat1.size(); i++) {
        std::vector<int> tmp_matches;
        for (int j = 0; j < match_rank1[i].size(); j++) {
            int tmp_match_idx = match_rank1[i][j];
            if (std::find(match_rank2[tmp_match_idx].begin(), match_rank2[tmp_match_idx].end(), i) != match_rank2[tmp_match_idx].end())
                tmp_matches.push_back(tmp_match_idx);
        }
        match_idx.push_back(tmp_matches);
    }

    // DEBUG
    if (ddd_verbose) {
        for (int i = 0; i < feat1.size(); i++) {
            std::cout << i << " | ";
            for (int j = 0; j < match_idx[i].size(); j++)
                std::cout << match_idx[i][j] << " ";
            std::cout << std::endl;
        }
    }

    // Compute Rt transform from second to first point cloud (k-ransac)
    tic();
    int num_inliers = ransacfitRt(world_keypoints1, world_keypoints2, match_idx, ransac_k, max_ransac_iter, ransac_thresh, Rt, ddd_verbose);
    std::cout << "Estimating rigid transform (via RANSAC-k). ";
    toc();
}

void ddd_compute_feature_cloud(float* scene_tsdf, int x_dim, int y_dim, int z_dim, float world_origin_x, float world_origin_y, float world_origin_z,
    float voxelSize,
    std::vector<std::vector<float>> &world_keypoints_out, std::vector<std::vector<float>> &feat_out) {
    
    // Find keypoints in TUDF
    tic();
    std::vector<std::vector<int>> keypoints = detect_keypoints(scene_tsdf, x_dim, y_dim, z_dim, 0.2f, 1.0f, 5, 100.0f);
    std::cout << "Detecting harris keypoints in TSDF volume. ";
    toc();

    // Filter out keypoints too close to the bounding box of the voxel volume
    int local_patch_radius = 15;
    std::vector<std::vector<int>> valid_keypoints;
    for (int i = 0; i < keypoints.size(); i++)
        if (keypoints[i][0] - local_patch_radius >= 0 && keypoints[i][0] + local_patch_radius < x_dim &&
            keypoints[i][1] - local_patch_radius >= 0 && keypoints[i][1] + local_patch_radius < y_dim &&
            keypoints[i][2] - local_patch_radius >= 0 && keypoints[i][2] + local_patch_radius < z_dim)
            valid_keypoints.push_back(keypoints[i]);

    // Compute ddd features from keypoints
    tic();
    feat_out = ddd_get_keypoint_feat(scene_tsdf, x_dim, y_dim, z_dim, valid_keypoints, local_patch_radius, ddd_verbose);
    std::cout << "Computing features for keypoints from TSDF volume. ";
    toc();

    // Convert valid keypoints from grid to world coordinates
    world_keypoints_out.clear();
    for (int i = 0; i < valid_keypoints.size(); i++) {
        std::vector<float> tmp_keypoint;
        tmp_keypoint.push_back((float)valid_keypoints[i][0] * voxelSize + world_origin_x);
        tmp_keypoint.push_back((float)valid_keypoints[i][1] * voxelSize + world_origin_y);
        tmp_keypoint.push_back((float)valid_keypoints[i][2] * voxelSize + world_origin_z);
        world_keypoints_out.push_back(tmp_keypoint);
    }
}

///////////////////////////////////////////////////////////////////////
void align2tsdf(float* scene_tsdf1, int x_dim1, int y_dim1, int z_dim1, float world_origin1_x, float world_origin1_y, float world_origin1_z,
                float* scene_tsdf2, int x_dim2, int y_dim2, int z_dim2, float world_origin2_x, float world_origin2_y, float world_origin2_z, 
                float voxelSize, float k_match_score_thresh, float ransac_k, float max_ransac_iter, float ransac_thresh, float* Rt) {

  std::vector<std::vector<float>> world_keypoints1, world_keypoints2;
  std::vector<std::vector<float>> feat1, feat2;

  ddd_compute_feature_cloud(scene_tsdf1, x_dim1, y_dim1, z_dim1, world_origin1_x, world_origin1_y, world_origin1_z, voxelSize, world_keypoints1, feat1);
  ddd_compute_feature_cloud(scene_tsdf2, x_dim2, y_dim2, z_dim2, world_origin2_x, world_origin2_y, world_origin2_z, voxelSize, world_keypoints2, feat2);

  std::vector<std::vector<float>> score_matrix1, score_matrix2;
  ddd_compare_feature_cloud(feat1, &score_matrix1, feat2, &score_matrix2);

  ddd_align_feature_cloud(world_keypoints1, feat1, score_matrix1, world_keypoints2, feat2, score_matrix2, voxelSize, k_match_score_thresh, ransac_k, max_ransac_iter, ransac_thresh, Rt);
}