
#include "pc2tsdf.h"

class FragmentMatcher
{
public:
    struct KeypointMatch
    {
        vec3f posA;
        vec3f posB;
        float alignmentError;
    };

    struct Result
    {
        void saveASCII(const std::string &filename) const
        {
            std::ofstream file(filename);
            file << "matchCount " << matches.size() << std::endl;
            file << "keypointsACount " << keypointsA.size() << std::endl;
            file << "keypointsBCount " << keypointsB.size() << std::endl;
            file << "transform" << std::endl;
            file << transformBToA;
            
            file << "#matches" << std::endl;
            for (const auto &match : matches)
            {
                file << match.posA << " " << match.posB << " " << match.alignmentError << std::endl;
            }
            file << "#keypointsA" << std::endl;
            for (vec3f v : keypointsA)
                file << v << std::endl;
            file << "#keypointsB" << std::endl;
            for (vec3f v : keypointsB)
                file << v << std::endl;
        }
        bool matchFound;

        mat4f transformBToA;
        
        std::vector<vec3f> keypointsA;
        std::vector<vec3f> keypointsB;
        std::vector<KeypointMatch> matches;
    };

    static void mulMatrix(const float m1[16], const float m2[16], float mOut[16]) {
      mOut[0]  = m1[0] * m2[0]  + m1[1] * m2[4]  + m1[2] * m2[8]   + m1[3] * m2[12];
      mOut[1]  = m1[0] * m2[1]  + m1[1] * m2[5]  + m1[2] * m2[9]   + m1[3] * m2[13];
      mOut[2]  = m1[0] * m2[2]  + m1[1] * m2[6]  + m1[2] * m2[10]  + m1[3] * m2[14];
      mOut[3]  = m1[0] * m2[3]  + m1[1] * m2[7]  + m1[2] * m2[11]  + m1[3] * m2[15];

      mOut[4]  = m1[4] * m2[0]  + m1[5] * m2[4]  + m1[6] * m2[8]   + m1[7] * m2[12];
      mOut[5]  = m1[4] * m2[1]  + m1[5] * m2[5]  + m1[6] * m2[9]   + m1[7] * m2[13];
      mOut[6]  = m1[4] * m2[2]  + m1[5] * m2[6]  + m1[6] * m2[10]  + m1[7] * m2[14];
      mOut[7]  = m1[4] * m2[3]  + m1[5] * m2[7]  + m1[6] * m2[11]  + m1[7] * m2[15];

      mOut[8]  = m1[8] * m2[0]  + m1[9] * m2[4]  + m1[10] * m2[8]  + m1[11] * m2[12];
      mOut[9]  = m1[8] * m2[1]  + m1[9] * m2[5]  + m1[10] * m2[9]  + m1[11] * m2[13];
      mOut[10] = m1[8] * m2[2]  + m1[9] * m2[6]  + m1[10] * m2[10] + m1[11] * m2[14];
      mOut[11] = m1[8] * m2[3]  + m1[9] * m2[7]  + m1[10] * m2[11] + m1[11] * m2[15];

      mOut[12] = m1[12] * m2[0] + m1[13] * m2[4] + m1[14] * m2[8]  + m1[15] * m2[12];
      mOut[13] = m1[12] * m2[1] + m1[13] * m2[5] + m1[14] * m2[9]  + m1[15] * m2[13];
      mOut[14] = m1[12] * m2[2] + m1[13] * m2[6] + m1[14] * m2[10] + m1[15] * m2[14];
      mOut[15] = m1[12] * m2[3] + m1[13] * m2[7] + m1[14] * m2[11] + m1[15] * m2[15];
    }

    static vec3f makeVec3(const std::vector<float> &v)
    {
        return vec3f(v[0], v[1], v[2]);
    }

    static Result match(const std::string &pointCloudFileA, const std::string &pointCloudFileB, int cloudIndA, int cloudIndB, float voxelSize, float truncationRadius, float maxKeypointMatchDist)
    {
        std::cout << std::endl << "Matching " << pointCloudFileA << " against " << pointCloudFileB << std::endl;
        
        const float k_match_score_thresh = 0.01f;
        const float ransac_k = 10; // RANSAC over top-k > k_match_score_thresh
        const float max_ransac_iter = 1000000;
        const float ransac_inlier_thresh = 0.04f;

        /*tic();
        FlatTSDF tsdfA = plyToTSDF(pointCloudFileA, voxelSize, truncationRadius);
        FlatTSDF tsdfB = plyToTSDF(pointCloudFileB, voxelSize, truncationRadius);
        std::cout << "Loading point clouds as TSDFs. ";
        toc();
        // std::cout << "OriginA: " << tsdfA.origin << std::endl;
        // std::cout << "DimA: " << tsdfA.dim << std::endl;

        ///////////////////////////////////////////////////////////////////

        float* Rt = new float[16]; // Contains rigid transform matrix
        Rt[12] = 0; Rt[13] = 0; Rt[14] = 0; Rt[15] = 1;
        align2tsdf(
            tsdfA.data.data(), tsdfA.dim.x, tsdfA.dim.y, tsdfA.dim.z, tsdfA.origin.x, tsdfA.origin.y, tsdfA.origin.z,
            tsdfB.data.data(), tsdfB.dim.x, tsdfB.dim.y, tsdfB.dim.z, tsdfB.origin.x, tsdfB.origin.y, tsdfB.origin.z,
            voxelSize, k_match_score_thresh, ransac_k, max_ransac_iter, ransac_inlier_thresh, Rt);*/

        FeatureCloud cloudA, cloudB;

        tic();
        plyToFeatureCloud(pointCloudFileA, voxelSize, truncationRadius, cloudA);
        plyToFeatureCloud(pointCloudFileB, voxelSize, truncationRadius, cloudB);
        toc();
        std::cout << "feature clouds loaded, aligning... " << std::endl;
        float* Rt = new float[16]; // Contains rigid transform matrix
        Rt[12] = 0; Rt[13] = 0; Rt[14] = 0; Rt[15] = 1;

        // for (int i = 0; i < cloudA.features.size(); i++) {
        //   for (int j = 0; j < 2048; j++) {
        //     std::cout << cloudA.features[i][j] << std::endl;
        //   }
        // }

        ddd_align_feature_cloud(cloudA.keypoints, cloudA.features, cloudB.keypoints, cloudB.features,
            voxelSize, k_match_score_thresh, ransac_k, max_ransac_iter, ransac_inlier_thresh, Rt);

        ///////////////////////////////////////////////////////////////////
        
        float prior_icp_avg_dist = 0;
        Result prior_icp_result;
        for (auto &x : cloudA.keypoints)
            prior_icp_result.keypointsA.push_back(makeVec3(x));
        for (auto &x : cloudB.keypoints)
            prior_icp_result.keypointsB.push_back(makeVec3(x));
        prior_icp_result.transformBToA = mat4f::identity();
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 3; j++)
                prior_icp_result.transformBToA(j, i) = Rt[j * 4 + i];
        pc2tsdf::UniformAccelerator acceleratorA(prior_icp_result.keypointsA, maxKeypointMatchDist);
        //transform B's keypoints into A's
        std::vector<vec3f> keypointsBtransformed = prior_icp_result.keypointsB;
        for (auto &bPt : keypointsBtransformed)
        {
            const vec3f bPtInA = prior_icp_result.transformBToA * bPt;
            const auto closestPt = acceleratorA.findClosestPoint(bPtInA);
            const float dist = vec3f::dist(bPtInA, closestPt.first);
            if (dist <= maxKeypointMatchDist)
            {
                KeypointMatch match;
                match.posA = closestPt.first;
                match.posB = bPt;
                match.alignmentError = dist;
                prior_icp_result.matches.push_back(match);
                prior_icp_avg_dist += dist;
            }
        }
        prior_icp_result.matchFound = prior_icp_result.matches.size() > 0;
        prior_icp_avg_dist = prior_icp_avg_dist/((float) prior_icp_result.matches.size());
        Result result = prior_icp_result;

        ///////////////////////////////////////////////////////////////////

        float* final_Rt = new float[16];
        final_Rt = Rt;

        // DISABLE ICP FOR NOW (too slow)
        bool use_matlab_icp = true;
        if (use_matlab_icp) {
          tic();
          // Create random hash ID for icp instance
          std::string instance_id = gen_rand_str(16);
          std::string pointcloud_filename1 = "TMPpointcloud1_" + instance_id + ".txt";
          std::string pointcloud_filename2 = "TMPpointcloud2_" + instance_id + ".txt";
          std::string icprt_filename = "TMPicpRt_" + instance_id + ".txt";

          // Save point clouds to files for matlab to read
          auto cloud1 = PointCloudIOf::loadFromFile(pointCloudFileA);
          FILE *fp = fopen(pointcloud_filename1.c_str(), "w");
          for (int i = 0; i < cloud1.m_points.size(); i++)
            fprintf(fp, "%f %f %f\n", cloud1.m_points[i].x, cloud1.m_points[i].y, cloud1.m_points[i].z);
          fclose(fp);
          auto cloud2 = PointCloudIOf::loadFromFile(pointCloudFileB);
          fp = fopen(pointcloud_filename2.c_str(), "w");
          for (int i = 0; i < cloud2.m_points.size(); i++) {
            vec3f tmp_point;
            tmp_point.x = Rt[0] * cloud2.m_points[i].x + Rt[1] * cloud2.m_points[i].y + Rt[2] * cloud2.m_points[i].z + Rt[3];
            tmp_point.y = Rt[4] * cloud2.m_points[i].x + Rt[5] * cloud2.m_points[i].y + Rt[6] * cloud2.m_points[i].z + Rt[7];
            tmp_point.z = Rt[8] * cloud2.m_points[i].x + Rt[9] * cloud2.m_points[i].y + Rt[10] * cloud2.m_points[i].z + Rt[11];
            fprintf(fp, "%f %f %f\n", tmp_point.x, tmp_point.y, tmp_point.z);
          }
          fclose(fp);

          std::string matlab_icp_filename = "matlab/main.m";
          std::string new_matlab_icp_filename = "matlab/main_" + instance_id + ".m";
          json_data_location_replace(matlab_icp_filename, new_matlab_icp_filename, ".txt", "_" + instance_id + ".txt");

          // Run matlab ICP
          sys_command("cd matlab; matlab -nojvm < main_" + instance_id + ".m >/dev/null; cd ..");
          float *icp_Rt = new float[16];
          int iret;
          fp = fopen(icprt_filename.c_str(), "r");
          for (int i = 0; i < 16; i++) {
            iret = fscanf(fp, "%f", &icp_Rt[i]);
          }
          fclose(fp);

          // Apply ICP Rt to current Rt
          mulMatrix(icp_Rt, Rt, final_Rt);

          delete [] icp_Rt;
          sys_command("rm " + pointcloud_filename1);
          sys_command("rm " + pointcloud_filename2);
          sys_command("rm " + icprt_filename);
          sys_command("rm " + new_matlab_icp_filename);

          // Double check to see if ICP is better
          float post_icp_avg_dist = 0;
          Result post_icp_result; 
          for (auto &x : cloudA.keypoints)
              post_icp_result.keypointsA.push_back(makeVec3(x));
          for (auto &x : cloudB.keypoints)
              post_icp_result.keypointsB.push_back(makeVec3(x));
          post_icp_result.transformBToA = mat4f::identity();
          for (int i = 0; i < 4; i++)
              for (int j = 0; j < 3; j++)
                  post_icp_result.transformBToA(j, i) = final_Rt[j * 4 + i];
          pc2tsdf::UniformAccelerator acceleratorA(post_icp_result.keypointsA, maxKeypointMatchDist);
          std::vector<vec3f> keypointsBtransformed = post_icp_result.keypointsB;
          for (auto &bPt : keypointsBtransformed)
          {
              const vec3f bPtInA = post_icp_result.transformBToA * bPt;
              const auto closestPt = acceleratorA.findClosestPoint(bPtInA);
              const float dist = vec3f::dist(bPtInA, closestPt.first);
              if (dist <= maxKeypointMatchDist)
              {
                  KeypointMatch match;
                  match.posA = closestPt.first;
                  match.posB = bPt;
                  match.alignmentError = dist;
                  post_icp_result.matches.push_back(match);
                  post_icp_avg_dist += dist;
              }
          }
          post_icp_result.matchFound = post_icp_result.matches.size() > 0;
          post_icp_avg_dist = post_icp_avg_dist/((float) post_icp_result.matches.size());

          printf("Pre-ICP avg err: %f\n", prior_icp_avg_dist);
          printf("Post-ICP avg err: %f\n", post_icp_avg_dist);
          if (post_icp_avg_dist >= prior_icp_avg_dist) {
            std::cout << "Using ICP to re-adjust RANSAC rigid transform. ";
            result = post_icp_result;
          } else {
            std::cout << "ICP results do not improve rigid transform, using RANSAC only.";
            final_Rt = Rt;
          }
          toc();
        }

        const bool debugDump = true;
        if (debugDump) {
            ///////////////////////////////////////////////////////////////////
            // DEBUG: save point aligned point clouds
            tic();

            auto cloud1 = PointCloudIOf::loadFromFile(pointCloudFileA);
            auto cloud2 = PointCloudIOf::loadFromFile(pointCloudFileB);

            // Rotate B points into A using final_Rt
            for (int i = 0; i < cloud2.m_points.size(); i++) {
                vec3f tmp_point;
                tmp_point.x = final_Rt[0] * cloud2.m_points[i].x + final_Rt[1] * cloud2.m_points[i].y + final_Rt[2] * cloud2.m_points[i].z + final_Rt[3];
                tmp_point.y = final_Rt[4] * cloud2.m_points[i].x + final_Rt[5] * cloud2.m_points[i].y + final_Rt[6] * cloud2.m_points[i].z + final_Rt[7];
                tmp_point.z = final_Rt[8] * cloud2.m_points[i].x + final_Rt[9] * cloud2.m_points[i].y + final_Rt[10] * cloud2.m_points[i].z + final_Rt[11];
                cloud2.m_points[i] = tmp_point;
            }

            // Make point clouds colorful
            ml::vec4f color1;
            for (int i = 0; i < 3; i++)
                color1[i] = gen_random_float(0.0, 1.0);
            for (int i = 0; i < cloud1.m_points.size(); i++)
                cloud1.m_colors[i] = color1;
            ml::vec4f color2;
            for (int i = 0; i < 3; i++)
                color2[i] = gen_random_float(0.0, 1.0);
            for (int i = 0; i < cloud2.m_points.size(); i++)
                cloud2.m_colors[i] = color2;

            // Save point clouds to file
            std::string pcfile1 = "results/debug" + std::to_string(cloudIndA) + "_" + std::to_string(cloudIndB) + "_" + std::to_string(cloudIndA) + ".ply";
            PointCloudIOf::saveToFile(pcfile1, cloud1);
            std::string pcfile2 = "results/debug" + std::to_string(cloudIndA) + "_" + std::to_string(cloudIndB) + "_" + std::to_string(cloudIndB) + ".ply";
            PointCloudIOf::saveToFile(pcfile2, cloud2);

            std::cout << "Saving point cloud visualizations. ";
            toc();
        }

        ///////////////////////////////////////////////////////////////////

        // // DEBUG: Use MATLAB RANSAC
        // sys_command("cd matlab; matlab -nojvm < main.m; cd ..");
        // float *tmp_matlab_rt = new float[12];
        // int iret;
        // FILE *fp = fopen("TMPrt.txt", "r");
        // for (int i = 0; i < 12; i++) {
        //   iret = fscanf(fp, "%f", &tmp_matlab_rt[i]);
        //   std::cout << tmp_matlab_rt[i] << std::endl;
        // }
        // fclose(fp);

        // if (debugDump) {
        //     // DEBUG: Use MATLAB RANSAC
        //     auto cloud1 = PointCloudIOf::loadFromFile(pointCloudFileA);
        //     auto cloud2 = PointCloudIOf::loadFromFile(pointCloudFileB);

        //     // Rotate B points into A using Rt
        //     for (int i = 0; i < cloud2.m_points.size(); i++) {
        //         vec3f tmp_point;
        //         tmp_point.x = tmp_matlab_rt[0] * cloud2.m_points[i].x + tmp_matlab_rt[1] * cloud2.m_points[i].y + tmp_matlab_rt[2] * cloud2.m_points[i].z;
        //         tmp_point.y = tmp_matlab_rt[4] * cloud2.m_points[i].x + tmp_matlab_rt[5] * cloud2.m_points[i].y + tmp_matlab_rt[6] * cloud2.m_points[i].z;
        //         tmp_point.z = tmp_matlab_rt[8] * cloud2.m_points[i].x + tmp_matlab_rt[9] * cloud2.m_points[i].y + tmp_matlab_rt[10] * cloud2.m_points[i].z;
        //         tmp_point.x = tmp_point.x + tmp_matlab_rt[3];
        //         tmp_point.y = tmp_point.y + tmp_matlab_rt[7];
        //         tmp_point.z = tmp_point.z + tmp_matlab_rt[11];
        //         cloud2.m_points[i] = tmp_point;
        //     }

        //     // Make point clouds colorful
        //     ml::vec4f color1;
        //     for (int i = 0; i < 3; i++)
        //         color1[i] = gen_random_float(0.0, 1.0);
        //     for (int i = 0; i < cloud1.m_points.size(); i++)
        //         cloud1.m_colors[i] = color1;
        //     ml::vec4f color2;
        //     for (int i = 0; i < 3; i++)
        //         color2[i] = gen_random_float(0.0, 1.0);
        //     for (int i = 0; i < cloud2.m_points.size(); i++)
        //         cloud2.m_colors[i] = color2;

        //     // Save point clouds to file
        //     std::string pcfile1 = "results/debug_matlab_" + std::to_string(cloudIndA) + "_" + std::to_string(cloudIndB) + "_" + std::to_string(cloudIndA) + ".ply";
        //     PointCloudIOf::saveToFile(pcfile1, cloud1);
        //     std::string pcfile2 = "results/debug_matlab_" + std::to_string(cloudIndA) + "_" + std::to_string(cloudIndB) + "_" + std::to_string(cloudIndB) + ".ply";
        //     PointCloudIOf::saveToFile(pcfile2, cloud2);
        // }

        ///////////////////////////////////////////////////////////////////

        // TODO: this is redundant with align2tsdf
        

        std::cout << "Keypoint matches found: " << result.matches.size() << std::endl;

        ///////////////////////////////////////////////////////////////////


        return result;
    }

private:
    
    struct FeatureCloud
    {
        std::vector<std::vector<float>> keypoints;
        std::vector<std::vector<float>> features;
    };

    struct FlatTSDF
    {
        std::vector<vec3f> makeKeypoints()
        {
            auto keypointsInt = detect_keypoints_filtered(data.data(), dim.x, dim.y, dim.z);
            std::vector<vec3f> result;
            for (auto &x : keypointsInt)
                result.push_back(vec3f(x[0], x[1], x[2]) * voxelSize + origin);
            return result;
        }
        vec3i dim;
        vec3f origin;
        float voxelSize;
        std::vector<float> data;
    };

    static void plyToFeatureCloud(const std::string &filename, float voxelSize, float truncationRadius, FeatureCloud &cloudOut)
    {
        const std::string cacheDir = "featCache/";

        sys_command("mkdir " + cacheDir);

        const std::string filenameBase = util::remove(util::split(filename, '/').back(), ".ply");
        const std::string keypointsFilename = cacheDir + filenameBase + "_keypoints.dat";
        const std::string featuresFilename = cacheDir + filenameBase + "_features.dat";

        if (!util::fileExists(keypointsFilename) || !util::fileExists(featuresFilename))
        {
            std::cout << "creating feature cloud for " << filename << std::endl;
            FlatTSDF tsdf = plyToTSDF(filename, voxelSize, truncationRadius);

            ddd_compute_feature_cloud(tsdf.data.data(), tsdf.dim.x, tsdf.dim.y, tsdf.dim.z, tsdf.origin.x, tsdf.origin.y, tsdf.origin.z, voxelSize,
                cloudOut.keypoints, cloudOut.features);

            saveGrid(keypointsFilename, cloudOut.keypoints);
            saveGrid(featuresFilename, cloudOut.features);
        }

        std::cout << "loading cached feature cloud for " << filename << std::endl;
        loadGrid(keypointsFilename, cloudOut.keypoints);
        loadGrid(featuresFilename, cloudOut.features);
    }

    static FlatTSDF plyToTSDF(const std::string &filename, float voxelSize, float truncationRadius)
    {
        auto cloud = PointCloudIOf::loadFromFile(filename);
        //std::cout << "Cloud size: " << cloud.m_points.size() << ", p5: " << cloud.m_points[5] << std::endl;
        pc2tsdf::TSDF tsdf;
        pc2tsdf::makeTSDF(cloud, voxelSize, truncationRadius, tsdf);
        
        // Convert TUDF grid data to float array
        FlatTSDF result;
        const vec3i dim = tsdf.data.getDimensions();
        result.dim = dim;
        result.data = std::vector<float>(dim.x * dim.y * dim.z);
        for (float &x : result.data)
            x = 1.0f;
        result.origin = tsdf.worldOrigin;
        result.voxelSize = voxelSize;
        
        for (int z = 0; z < dim.z; z++)
            for (int y = 0; y < dim.y; y++)
                for (int x = 0; x < dim.x; x++)
                    result.data[z * dim.y * dim.x + y * dim.x + x] = tsdf.data(x, y, z) / truncationRadius;
        return result;
    }

    static void saveGrid(const std::string &filename, std::vector<std::vector<float>> &grid)
    {
        const size_t dimX = grid.size(), dimY = grid[0].size();
        Grid2f g(dimX, dimY);
        for (int x = 0; x < dimX; x++)
            for (int y = 0; y < dimY; y++)
                g(x, y) = grid[x][y];

        FILE *file = util::checkedFOpen(filename.c_str(), "wb");
        util::checkedFWrite(&dimX, sizeof(size_t), 1, file);
        util::checkedFWrite(&dimY, sizeof(size_t), 1, file);
        util::checkedFWrite(g.getData(), sizeof(float), dimX * dimY, file);
        fclose(file);
    }

    static void loadGrid(const std::string &filename, std::vector<std::vector<float>> &grid)
    {
        size_t dimX, dimY;
        
        FILE *file = util::checkedFOpen(filename.c_str(), "rb");
        util::checkedFRead(&dimX, sizeof(size_t), 1, file);
        util::checkedFRead(&dimY, sizeof(size_t), 1, file);

        Grid2f g(dimX, dimY);
        util::checkedFRead(g.getData(), sizeof(float), dimX * dimY, file);
        fclose(file);

        grid.resize(dimX);
        for (auto &x : grid)
            x.resize(dimY);

        for (int x = 0; x < dimX; x++)
            for (int y = 0; y < dimY; y++)
                grid[x][y] = g(x, y);

    }
};