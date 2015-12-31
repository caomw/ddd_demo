// #include "pc2tsdf/pc2tsdf.h"
// #include "detect_keypoints.h"
// #include "ddd.h"
// #include "fragmentMatcher/fragmentMatcher.h"
#include <vector>
#include <numeric>      
#include <algorithm>   
#include <iostream>
#include <fstream> 
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cmath>
#include "cluster_ransacK.cpp"
#include <functional>

bool fileExists(const std::string& filename) {
    std::ifstream file(filename.c_str());
    return (!file.fail());
}

bool sort_arr_desc_compare(int a, int b, float* data) {
    return data[a]>data[b];
}

void ddd_align_feature_cloud(const std::vector<std::vector<float>> &world_keypoints1, std::vector<std::vector<float>> &feat1, const std::vector<std::vector<float>> &score_matrix1,
                             const std::vector<std::vector<float>> &world_keypoints2, std::vector<std::vector<float>> &feat2, const std::vector<std::vector<float>> &score_matrix2,
                             float voxelSize, float k_match_score_thresh, float ransac_k, float max_ransac_iter, float ransac_thresh, float* Rt) {

    // For each keypoint from first set, find indices of all keypoints
    // in second set with score > k_match_score_thresh
    std::vector<std::vector<int>> match_rank1;
    for (int i = 0; i < feat1.size(); i++) {
        // Sort score vector in descending fashion
        std::vector<float> tmp_score_vect = score_matrix1[i];
        float* tmp_score_vect_arr = &tmp_score_vect[0];
        int* tmp_score_idx = new int[tmp_score_vect.size()];
        std::iota(tmp_score_idx, tmp_score_idx + tmp_score_vect.size(), 0);
        std::sort(tmp_score_idx, tmp_score_idx + tmp_score_vect.size(), std::bind(sort_arr_desc_compare, std::placeholders::_1, std::placeholders::_2, tmp_score_vect_arr));
        std::vector<int> tmp_score_rank;
        for (int j = 0; j < feat2.size(); j++)
            if (tmp_score_vect_arr[tmp_score_idx[j]] > k_match_score_thresh)
                tmp_score_rank.push_back(tmp_score_idx[j]);
        // std::cout << tmp_score_rank.size() << std::endl;
        match_rank1.push_back(tmp_score_rank);
    }

    // For each keypoint from second set, find indices of all keypoints
    // in first set with score > k_match_score_thresh
    std::vector<std::vector<int>> match_rank2;
    for (int i = 0; i < feat2.size(); i++) {
        // Sort score vector in descending fashion
        std::vector<float> tmp_score_vect = score_matrix2[i];
        float* tmp_score_vect_arr = &tmp_score_vect[0];
        int* tmp_score_idx = new int[tmp_score_vect.size()];
        std::iota(tmp_score_idx, tmp_score_idx + tmp_score_vect.size(), 0);
        std::sort(tmp_score_idx, tmp_score_idx + tmp_score_vect.size(), std::bind(sort_arr_desc_compare, std::placeholders::_1, std::placeholders::_2, tmp_score_vect_arr));
        std::vector<int> tmp_score_rank;
        for (int j = 0; j < feat1.size(); j++)
            if (tmp_score_vect_arr[tmp_score_idx[j]] > k_match_score_thresh)
                tmp_score_rank.push_back(tmp_score_idx[j]);
        // std::cout << tmp_score_rank.size() << std::endl;
        match_rank2.push_back(tmp_score_rank);
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
    for (int i = 0; i < feat1.size(); i++) {
        std::cout << i << " | ";
        for (int j = 0; j < match_idx[i].size(); j++)
            std::cout << match_idx[i][j] << " ";
        std::cout << std::endl;
    }

    // Compute Rt transform from second to first point cloud (k-ransac)
    int num_inliers = ransacfitRt(world_keypoints1, world_keypoints2, match_idx, ransac_k, max_ransac_iter, ransac_thresh, Rt, true);
}

void loadGrid(std::string &filename, std::vector<std::vector<float>> &grid) {
    size_t dimX, dimY;
    
    if(!fileExists(filename)) {
        std::cout << "Could not read file: " << filename << std::endl;
        exit(1);
    }

    FILE *file = fopen(filename.c_str(), "rb");
    size_t dummy1 = fread(&dimX, sizeof(size_t), 1, file);
    size_t dummy2 = fread(&dimY, sizeof(size_t), 1, file);

    // Grid2f g(dimX, dimY);
    // util::checkedFRead(g.getData(), sizeof(float), dimX * dimY, file);

    
    std::cout << dimX << " " << dimY << std::endl;

    // grid.resize(dimX);
    for (int i = 0; i < dimX; i++) {
        // grid[i].resize(dimY);
        std:: vector<float> tmp_vec;
        grid.push_back(tmp_vec);
    }

    // for (int x = 0; x < dimX; x++) {
    //     // std:: vector<float> tmp_vec;
    //     // for (int y = 0; y < dimY; y++) {
    //     //     float tmp_val;
    //     //     fread(&tmp_val, sizeof(float), 1, file);
    //     //     tmp_vec.push_back(tmp_val);
    //     // }
    //     fread(&grid[x][0], sizeof(float), dimY, file);
    //     // grid.push_back(tmp_vec);
    // }

    for (int y = 0; y < dimY; y++) {
        for (int x = 0; x < dimX; x++) {
            float tmp_val;
            size_t dumm3 = fread(&tmp_val, sizeof(float), 1, file);
            grid[x].push_back(tmp_val);
        }
    }

    // for (int x = 0; x < dimX; x++) {
    //     for (int y = 0; y < dimY; y++) {
    //         std::cout << grid[x][y] << std::endl;
    //     }
    // }


    fclose(file);
            // std::cout << grid[x][y] << std::endl;
            // grid[x][y] = g(x, y);

}

void cluster_ransac(std::string fragment1_name, std::string fragment2_name, int fragment1_idx, int fragment2_idx) {
    
    // Load first fragment's keypoints
    std::vector<std::vector<float>> keypoints1;
    std::string keypoints1_filename = "featCache/cloud_bin_" + std::to_string((long long)fragment1_idx) + "_keypoints.dat";
    loadGrid(keypoints1_filename, keypoints1);

    // Load second fragment's keypoints
    std::vector<std::vector<float>> keypoints2;
    std::string keypoints2_filename = "featCache/cloud_bin_" + std::to_string((long long)fragment2_idx) + "_keypoints.dat";
    loadGrid(keypoints2_filename, keypoints2);

    // Load first fragment's features
    std::vector<std::vector<float>> feat1;
    std::string feat1_filename = "featCache/cloud_bin_" + std::to_string((long long)fragment1_idx) + "_features.dat";
    loadGrid(feat1_filename, feat1);

    // Load second fragment's features
    std::vector<std::vector<float>> feat2;
    std::string feat2_filename = "featCache/cloud_bin_" + std::to_string((long long)fragment2_idx) + "_features.dat";
    loadGrid(feat2_filename, feat2);

    // Load first score matrix
    std::vector<std::vector<float>> score_matrix1;
    std::string score_matrix1_filename = "featCache/match_scores_" + std::to_string((long long)fragment1_idx) + "_" + std::to_string((long long)fragment2_idx) + ".dat";
    loadGrid(score_matrix1_filename, score_matrix1);

    // Load second score matrix
    std::vector<std::vector<float>> score_matrix2;
    std::string score_matrix2_filename = "featCache/match_scores_" + std::to_string((long long)fragment2_idx) + "_" + std::to_string((long long)fragment1_idx) + ".dat";
    loadGrid(score_matrix2_filename, score_matrix2);

    float* Rt = new float[16]; // Contains rigid transform matrix
    Rt[12] = 0; Rt[13] = 0; Rt[14] = 0; Rt[15] = 1;

    const float k_match_score_thresh = 0.1f;
    const float ransac_k = 10; // RANSAC over top-k > k_match_score_thresh
    const float max_ransac_iter = 1000000;
    const float ransac_inlier_thresh = 0.04f;
    const float voxelSize = 0.01f;

    // for (int i = 0; i < keypoints1.size(); i++)
    //     std::cout << keypoints1[i][0] << " " << keypoints1[i][1] << " " << keypoints1[i][2] << std::endl;

    ddd_align_feature_cloud(keypoints1, feat1, score_matrix1, keypoints2, feat2, score_matrix2, voxelSize, k_match_score_thresh, ransac_k, max_ransac_iter, ransac_inlier_thresh, Rt);

    

    // Save to file in results
    std::string rt_filename = "results/rt_" + std::to_string((long long)fragment1_idx) + "_" + std::to_string((long long)fragment2_idx) + ".dat";
    FILE *file = fopen(rt_filename.c_str(), "wb");
    for (int i = 0; i < 4; i++)
        fprintf(file, "%f %f %f %f\n", Rt[4*i+0], Rt[4*i+1], Rt[4*i+2], Rt[4*i+3]);
    fclose(file);

}

int main(int argc, char **argv) {
    const float voxelSize = 0.01f;
    const float truncationRadius = 0.05f;
    const float maxKeypointMatchDist = 0.04f;
    const int fragmentCount = 57;
    //const string fragmentPrefix = "/data/andyz/kinfu/data/augICLNUIMDataset/fragments/livingroom1-fragments-ply/cloud_bin_";
    const std::string fragmentPrefix = "../ddd_data/cloud_bin_";

    srand ((unsigned int)time(NULL));

    std::vector<std::string> allFragments;
    
    for (int i = 0; i < fragmentCount; i++)
    {
        const std::string fragmentFilename = fragmentPrefix + std::to_string((long long)i) + ".ply";
        std::cout << fragmentFilename << std::endl;
        if (!fileExists(fragmentFilename))
        {
            std::cout << "file not found: " << fragmentFilename << std::endl;
            return -1;
        }
        allFragments.push_back(fragmentFilename);
    }
    
    bool dir_success = system("mkdir results");

    // For running on cpu clusters
    int idx = std::atoi(argv[1]);
    int i = (int) std::floor(((float)idx)/((float)fragmentCount));
    int j = idx%fragmentCount;
    const std::string resultFilename = "results/match" + std::to_string((long long)i) + "-" + std::to_string((long long)j) + ".txt";
    if (j <= i || fileExists(resultFilename))
        return 0;
    cluster_ransac(allFragments[i], allFragments[j], i, j);
    // auto result = FragmentMatcher::match(allFragments[i], allFragments[j], i, j, voxelSize, truncationRadius, maxKeypointMatchDist);
    // result.saveASCII(resultFilename);

    return 0;
}