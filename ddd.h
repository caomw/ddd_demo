#include <vector>

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

///////////////////////////////////////////////////////////////////////
void sys_command(std::string str) {
  if (system(str.c_str()))
    return;
}

///////////////////////////////////////////////////////////////////////
std::vector<std::vector<float>> ddd_get_keypoint_feat(float* volume, int x_dim, int y_dim, int z_dim, std::vector<std::vector<int>> &keypoints, int patch_radius) {

  // Init tensor files for marvin
  std::string data_tensor_filename = "TMPdata.tensor";
  std::string label_tensor_filename = "TMPlabels.tensor";
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
    std::cout << "Iteration " << i + 1 << "/" << num_keypoints << ": " << keypoints[i][0] << " " << keypoints[i][1] << " " << keypoints[i][2] << std::endl;
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
  std::string feat_tensor_filename = "feat_response.tensor";
  std::string model_filename = "ddd/dddnet.marvin";
  std::string architecture_filename = "ddd/featnet.json";
  std::string marvin_filename = "ddd/marvin";
  std::string cuda_lib_dir = "/usr/local/cuda/lib64";

  // Run marvin to get tensor file of feature vectors
  sys_command("rm " + feat_tensor_filename);
  sys_command("export LD_LIBRARY_PATH=LD_LIBRARY_PATH:" + cuda_lib_dir + "; ./" + marvin_filename + " test " + architecture_filename + " " + model_filename + " feat " + feat_tensor_filename);
  sys_command("rm " + data_tensor_filename);
  sys_command("rm " + label_tensor_filename);

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

  return feat;
}

///////////////////////////////////////////////////////////////////////
std::vector<std::vector<float>> ddd_compare_feat(std::vector<std::vector<float>> &feat1, std::vector<std::vector<float>> &feat2) {

  int feat_dim = 2048;
  int num_cases = feat1.size() * feat2.size();

  // Init tensor files for marvin
  std::string data_tensor_filename = "TMPdata.tensor";
  std::string label_tensor_filename = "TMPlabels.tensor";
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
      std::cout << "Iteration " << i*feat2.size() + j << "/" << feat1.size() * feat2.size() - 1 << std::endl;
      // Save feature vector 2
      for (int k = 0; k < feat_dim; k++) {
        float feat2_val = feat2[j][k];
        fwrite(&feat2_val, sizeof(float), 1, data_tensor_fp);
      }
      // Save feature vector 1
      for (int k = 0; k < feat_dim; k++) {
        float feat1_val = feat1[i][k];
        fwrite(&feat1_val, sizeof(float), 1, data_tensor_fp);
      }
      float tmp_label = 0;
      fwrite(&tmp_label, sizeof(float), 1, label_tensor_fp);
    }
  }

  fclose(label_tensor_fp);
  fclose(data_tensor_fp);

  // Marvin params
  std::string prob_tensor_filename = "prob_response.tensor";
  std::string model_filename = "ddd/dddnet.marvin";
  std::string architecture_filename = "ddd/metricnet.json";
  std::string marvin_filename = "ddd/marvin";
  std::string cuda_lib_dir = "/usr/local/cuda/lib64";

  // Run marvin to get tensor file of match scores
  model_filename = "ddd/dddnet.marvin";
  sys_command("rm " + prob_tensor_filename);
  sys_command("export LD_LIBRARY_PATH=LD_LIBRARY_PATH:" + cuda_lib_dir + "; ./" + marvin_filename + " test " + architecture_filename + " " + model_filename + " prob " + prob_tensor_filename);
  sys_command("rm " + data_tensor_filename);
  sys_command("rm " + label_tensor_filename);

  // Save match scores to matrix 
  std::vector<std::vector<float>> score_matrix;
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
    score_matrix.push_back(tmp_score_vect);
  }

  sys_command("rm " + prob_tensor_filename);
  return score_matrix;
}